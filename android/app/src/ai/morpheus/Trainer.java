package ai.morpheus;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Morpheus learning on the phone: a Java port of morpheus/grad.py (the hand-written backward
 * pass and AdamW), checked gradient by gradient against it in tests/test_android.py.
 *
 * The float weights underneath ("latent") get the updates; the brain thinks with their quantized
 * version (Quant.fake), so a ternary brain stays ternary while it learns. Rows of a batch are
 * independent until the update, so they are spread over the phone's CPU cores.
 */
public final class Trainer {

    final Brain base;                        // config and rotary tables
    final List<String> names = new ArrayList<>();
    final Map<String, float[]> w = new HashMap<>(), m = new HashMap<>(), v = new HashMap<>();
    int t = 0;
    float weightDecay = 0.1f, beta1 = 0.9f, beta2 = 0.99f, gradClip = 1f;
    final int threads;

    /**
     * Adapters (LoRA): with rank > 0 the brain itself is frozen and what is learned lives in small
     * low-rank additions, W = Q(W0) + B A, on every attention and MLP matrix. A ternary brain's
     * skills (like copying a new word letter by letter) are then safe from the jumps that flipping
     * its -1/0/+1 weights would cause; the adapters of the nano brain are only ~33K numbers.
     */
    public int rank = 0;
    final Map<String, float[]> A = new HashMap<>(), Bm = new HashMap<>();
    final Map<String, float[]> mA = new HashMap<>(), vA = new HashMap<>(), mB = new HashMap<>(), vB = new HashMap<>();
    Map<String, float[]> frozen;  // Q(W0), computed once

    static boolean adapted(String name) {
        return name.startsWith("blocks.") && !name.endsWith("wte.weight");
    }

    /** Switch to learning in adapters of the given rank (fresh adapters start as zero change). */
    public void useAdapters(int r, long seed) {
        rank = r;
        java.util.Random rnd = new java.util.Random(seed);
        for (String k : names) {
            if (!adapted(k)) continue;
            int[] sh = base.shape.get(k);
            float[] a = new float[r * sh[1]];
            float std = (float) (1.0 / Math.sqrt(sh[1]));
            for (int i = 0; i < a.length; i++) a[i] = (float) (rnd.nextGaussian() * std);
            A.put(k, a);
            Bm.put(k, new float[sh[0] * r]);  // B = 0: the adapted brain starts identical
            mA.put(k, new float[a.length]); vA.put(k, new float[a.length]);
            mB.put(k, new float[sh[0] * r]); vB.put(k, new float[sh[0] * r]);
        }
    }

    public Trainer(Brain brain) {
        base = brain;
        names.addAll(new java.util.TreeSet<>(brain.w.keySet()));
        for (String k : names) {
            w.put(k, brain.w.get(k).clone());
            m.put(k, new float[brain.w.get(k).length]);
            v.put(k, new float[brain.w.get(k).length]);
        }
        threads = Math.max(1, Math.min(8, Runtime.getRuntime().availableProcessors()));
    }

    /** The weights the brain thinks with: quantized in the brain's own format (plus adapters). */
    public Map<String, float[]> effective() {
        Map<String, float[]> out = new HashMap<>();
        if (rank > 0 && frozen == null) {
            frozen = new HashMap<>();
            for (String k : names) {
                int[] sh = base.shape.get(k);
                frozen.put(k, Quant.fake(w.get(k), sh[0], sh[1], base.scheme.get(k)));
            }
        }
        for (String k : names) {
            int[] sh = base.shape.get(k);
            if (rank == 0) { out.put(k, Quant.fake(w.get(k), sh[0], sh[1], base.scheme.get(k))); continue; }
            float[] q = frozen.get(k);
            if (!A.containsKey(k)) { out.put(k, q); continue; }
            float[] merged = q.clone(), a = A.get(k), b = Bm.get(k);
            int rows = sh[0], cols = sh[1];
            for (int o = 0; o < rows; o++)
                for (int r = 0; r < rank; r++) {
                    float br = b[o * rank + r];
                    if (br == 0f) continue;
                    for (int i = 0; i < cols; i++) merged[o * cols + i] += br * a[r * cols + i];
                }
            out.put(k, merged);
        }
        return out;
    }

    public Brain toBrain() { return new Brain(base, effective()); }

    // ------------------------------------------------------------------ one row: forward, then backward

    static void rmsFwd(float[] x, int T, int d, float[] y, float[] r) {
        for (int t = 0; t < T; t++) {
            float ss = 0f;
            for (int i = 0; i < d; i++) ss += x[t * d + i] * x[t * d + i];
            r[t] = (float) (1.0 / Math.sqrt(ss / d + 1e-6f));
            for (int i = 0; i < d; i++) y[t * d + i] = x[t * d + i] * r[t];
        }
    }

    /** dx += r * (dy - y * mean(dy * y)) */
    static void rmsBwdAdd(float[] dy, float[] y, float[] r, int T, int d, float[] dx) {
        for (int t = 0; t < T; t++) {
            float dot = 0f;
            for (int i = 0; i < d; i++) dot += dy[t * d + i] * y[t * d + i];
            dot /= d;
            for (int i = 0; i < d; i++) dx[t * d + i] += r[t] * (dy[t * d + i] - y[t * d + i] * dot);
        }
    }

    /** out (T, nOut) = in (T, nIn) @ W^T, W row-major (nOut, nIn) */
    static void linear(float[] in, float[] W, float[] out, int T, int nIn, int nOut) {
        for (int t = 0; t < T; t++)
            for (int o = 0; o < nOut; o++) {
                float s = 0f;
                int row = o * nIn, at = t * nIn;
                for (int i = 0; i < nIn; i++) s += W[row + i] * in[at + i];
                out[t * nOut + o] = s;
            }
    }

    /** gW += dOut^T @ in, and (if dIn != null) dIn += dOut @ W */
    static void linearBwd(float[] dOut, float[] in, float[] W, float[] gW, float[] dIn, int T, int nIn, int nOut) {
        for (int t = 0; t < T; t++)
            for (int o = 0; o < nOut; o++) {
                float g = dOut[t * nOut + o];
                if (g == 0f) continue;
                int row = o * nIn, at = t * nIn;
                for (int i = 0; i < nIn; i++) {
                    gW[row + i] += g * in[at + i];
                    if (dIn != null) dIn[at + i] += g * W[row + i];
                }
            }
    }

    void rope(float[] x, int T, int heads, float sign) {
        int hd = base.hd, half = hd / 2;
        for (int t = 0; t < T; t++)
            for (int h = 0; h < heads; h++) {
                int off = (t * heads + h) * hd;
                for (int i = 0; i < half; i++) {
                    float c = base.cos[t][i], s = sign * base.sin[t][i];
                    float x1 = x[off + i], x2 = x[off + i + half];
                    x[off + i] = x1 * c - x2 * s;
                    x[off + i + half] = x1 * s + x2 * c;
                }
            }
    }

    /** Everything the backward pass needs about one layer application. */
    static final class Tape {
        float[] h, r1, q, k, v, att, y, h2, r2, relu, act;
    }

    /** Forward one row; returns logits (T, V) and fills the tapes (if tapes != null). */
    float[] forward(int[] ids, Map<String, float[]> W, List<Tape> tapes, float[][] finalNorm) {
        Brain b = base;
        int T = ids.length, d = b.nEmbd, H = b.nHead, K = b.nKv, hd = b.hd, F = b.hidden, V = b.vocab;
        int G = H / K;
        float scale = (float) (1.0 / Math.sqrt(hd));
        float[] x = new float[T * d], wte = W.get("wte.weight");
        for (int t = 0; t < T; t++) System.arraycopy(wte, ids[t] * d, x, t * d, d);
        for (int loop = 0; loop < b.nLoop; loop++)
            for (int l = 0; l < b.nLayer; l++) {
                String p = "blocks." + l + ".";
                Tape tp = new Tape();
                tp.h = new float[T * d];
                tp.r1 = new float[T];
                rmsFwd(x, T, d, tp.h, tp.r1);
                tp.q = new float[T * H * hd];
                float[] kv = new float[T * 2 * K * hd];
                linear(tp.h, W.get(p + "attn.q.weight"), tp.q, T, d, H * hd);
                linear(tp.h, W.get(p + "attn.kv.weight"), kv, T, d, 2 * K * hd);
                tp.k = new float[T * K * hd];
                tp.v = new float[T * K * hd];
                for (int t = 0; t < T; t++) {
                    System.arraycopy(kv, t * 2 * K * hd, tp.k, t * K * hd, K * hd);
                    System.arraycopy(kv, t * 2 * K * hd + K * hd, tp.v, t * K * hd, K * hd);
                }
                rope(tp.q, T, H, 1f);
                rope(tp.k, T, K, 1f);
                tp.att = new float[H * T * T];
                tp.y = new float[T * H * hd];
                for (int hh = 0; hh < H; hh++) {
                    int g = hh / G;
                    for (int t = 0; t < T; t++) {
                        int row = (hh * T + t) * T;
                        float max = Float.NEGATIVE_INFINITY;
                        for (int s = 0; s <= t; s++) {
                            float dot = 0f;
                            for (int i = 0; i < hd; i++) dot += tp.q[(t * H + hh) * hd + i] * tp.k[(s * K + g) * hd + i];
                            tp.att[row + s] = dot * scale;
                            max = Math.max(max, tp.att[row + s]);
                        }
                        float sum = 0f;
                        for (int s = 0; s <= t; s++) { tp.att[row + s] = (float) Math.exp(tp.att[row + s] - max); sum += tp.att[row + s]; }
                        for (int s = 0; s <= t; s++) tp.att[row + s] /= sum;
                        for (int i = 0; i < hd; i++) {
                            float acc = 0f;
                            for (int s = 0; s <= t; s++) acc += tp.att[row + s] * tp.v[(s * K + g) * hd + i];
                            tp.y[(t * H + hh) * hd + i] = acc;
                        }
                    }
                }
                float[] o = new float[T * d];
                linear(tp.y, W.get(p + "attn.o.weight"), o, T, H * hd, d);
                for (int i = 0; i < T * d; i++) x[i] += o[i];
                tp.h2 = new float[T * d];
                tp.r2 = new float[T];
                rmsFwd(x, T, d, tp.h2, tp.r2);
                tp.relu = new float[T * F];
                linear(tp.h2, W.get(p + "mlp.fc.weight"), tp.relu, T, d, F);
                tp.act = new float[T * F];
                for (int i = 0; i < T * F; i++) { tp.relu[i] = Math.max(tp.relu[i], 0f); tp.act[i] = tp.relu[i] * tp.relu[i]; }
                float[] mo = new float[T * d];
                linear(tp.act, W.get(p + "mlp.proj.weight"), mo, T, F, d);
                for (int i = 0; i < T * d; i++) x[i] += mo[i];
                if (tapes != null) tapes.add(tp);
            }
        float[] hf = new float[T * d], rf = new float[T];
        rmsFwd(x, T, d, hf, rf);
        if (finalNorm != null) { finalNorm[0] = hf; finalNorm[1] = rf; }
        float[] logits = new float[T * V];
        linear(hf, wte, logits, T, d, V);
        return logits;
    }

    /** Logits for every position of one row, with the given weights (no gradients). */
    public float[] logits(int[] ids, Map<String, float[]> W) { return forward(ids, W, null, null); }

    static void softmaxRows(float[] z, int T, int V) {
        for (int t = 0; t < T; t++) {
            float max = Float.NEGATIVE_INFINITY, sum = 0f;
            for (int i = 0; i < V; i++) max = Math.max(max, z[t * V + i]);
            for (int i = 0; i < V; i++) { z[t * V + i] = (float) Math.exp(z[t * V + i] - max); sum += z[t * V + i]; }
            for (int i = 0; i < V; i++) z[t * V + i] /= sum;
        }
    }

    /** Adds this row's gradients into g; returns the row's summed cross-entropy. */
    double rowGrads(int[] x, int[] y, float[] soft, float alpha, Map<String, float[]> W, Map<String, float[]> g, int N) {
        Brain b = base;
        int T = x.length, d = b.nEmbd, H = b.nHead, K = b.nKv, hd = b.hd, F = b.hidden, V = b.vocab;
        int G = H / K;
        float scale = (float) (1.0 / Math.sqrt(hd));
        List<Tape> tapes = new ArrayList<>();
        float[][] fin = new float[2][];
        float[] p = forward(x, W, tapes, fin);
        softmaxRows(p, T, V);
        double loss = 0;
        float[] dl = new float[T * V];
        for (int t = 0; t < T; t++) {
            loss -= Math.log(p[t * V + y[t]] + 1e-12);
            for (int i = 0; i < V; i++) {
                float ce = p[t * V + i] - (i == y[t] ? 1f : 0f);
                float grad = (soft != null && alpha > 0) ? (1 - alpha) * ce + alpha * (p[t * V + i] - soft[t * V + i]) : ce;
                dl[t * V + i] = grad / N;
            }
        }
        float[] wte = W.get("wte.weight"), gWte = g.get("wte.weight");
        float[] dhf = new float[T * d];
        linearBwd(dl, fin[0], wte, gWte, dhf, T, d, V);
        float[] dx = new float[T * d];
        rmsBwdAdd(dhf, fin[0], fin[1], T, d, dx);
        for (int li = tapes.size() - 1; li >= 0; li--) {
            Tape tp = tapes.get(li);
            String pre = "blocks." + (li % b.nLayer) + ".";
            // MLP: x_out = x_mid + relu(h2 Wfc^T)^2 Wproj^T
            float[] dact = new float[T * F];
            linearBwd(dx, tp.act, W.get(pre + "mlp.proj.weight"), g.get(pre + "mlp.proj.weight"), dact, T, F, d);
            for (int i = 0; i < T * F; i++) dact[i] *= 2f * tp.relu[i];
            float[] dh2 = new float[T * d];
            linearBwd(dact, tp.h2, W.get(pre + "mlp.fc.weight"), g.get(pre + "mlp.fc.weight"), dh2, T, d, F);
            rmsBwdAdd(dh2, tp.h2, tp.r2, T, d, dx);
            // attention: x_mid = x_in + y Wo^T
            float[] dy = new float[T * H * hd];
            linearBwd(dx, tp.y, W.get(pre + "attn.o.weight"), g.get(pre + "attn.o.weight"), dy, T, H * hd, d);
            float[] dq = new float[T * H * hd], dk = new float[T * K * hd], dv = new float[T * K * hd];
            float[] datt = new float[T];
            for (int hh = 0; hh < H; hh++) {
                int gk = hh / G;
                for (int t = 0; t < T; t++) {
                    int row = (hh * T + t) * T;
                    float dotSum = 0f;
                    for (int s = 0; s <= t; s++) {
                        float dot = 0f;
                        for (int i = 0; i < hd; i++) dot += dy[(t * H + hh) * hd + i] * tp.v[(s * K + gk) * hd + i];
                        datt[s] = dot;
                        dotSum += dot * tp.att[row + s];
                        float a = tp.att[row + s];
                        for (int i = 0; i < hd; i++) dv[(s * K + gk) * hd + i] += a * dy[(t * H + hh) * hd + i];
                    }
                    for (int s = 0; s <= t; s++) {
                        float ds = tp.att[row + s] * (datt[s] - dotSum) * scale;
                        if (ds == 0f) continue;
                        for (int i = 0; i < hd; i++) {
                            dq[(t * H + hh) * hd + i] += ds * tp.k[(s * K + gk) * hd + i];
                            dk[(s * K + gk) * hd + i] += ds * tp.q[(t * H + hh) * hd + i];
                        }
                    }
                }
            }
            rope(dq, T, H, -1f);  // the rotation's transpose is the rotation backwards
            rope(dk, T, K, -1f);
            float[] dkv = new float[T * 2 * K * hd];
            for (int t = 0; t < T; t++) {
                System.arraycopy(dk, t * K * hd, dkv, t * 2 * K * hd, K * hd);
                System.arraycopy(dv, t * K * hd, dkv, t * 2 * K * hd + K * hd, K * hd);
            }
            float[] dh = new float[T * d];
            linearBwd(dq, tp.h, W.get(pre + "attn.q.weight"), g.get(pre + "attn.q.weight"), dh, T, d, H * hd);
            linearBwd(dkv, tp.h, W.get(pre + "attn.kv.weight"), g.get(pre + "attn.kv.weight"), dh, T, d, 2 * K * hd);
            rmsBwdAdd(dh, tp.h, tp.r1, T, d, dx);
        }
        for (int t = 0; t < T; t++)
            for (int i = 0; i < d; i++) gWte[x[t] * d + i] += dx[t * d + i];
        return loss;
    }

    Map<String, float[]> zeros() {
        Map<String, float[]> g = new HashMap<>();
        for (String k : names) g.put(k, new float[w.get(k).length]);
        return g;
    }

    /** Mean loss and gradients for a batch; soft/alpha (per row) mix in a teacher's answers. */
    public Object[] lossAndGrads(final int[][] x, final int[][] y, final float[][] soft, final float[] alpha,
                                 final Map<String, float[]> W) throws Exception {
        final int B = x.length, N = B * x[0].length;
        int workers = Math.min(threads, B);
        List<Map<String, float[]>> parts = new ArrayList<>();
        final double[] losses = new double[B];
        if (workers == 1) {
            Map<String, float[]> g = zeros();
            for (int r = 0; r < B; r++) losses[r] = rowGrads(x[r], y[r], soft == null ? null : soft[r], alpha == null ? 0 : alpha[r], W, g, N);
            parts.add(g);
        } else {
            ExecutorService pool = Executors.newFixedThreadPool(workers);
            try {
                List<Future<Map<String, float[]>>> futures = new ArrayList<>();
                for (int wk = 0; wk < workers; wk++) {
                    final int first = wk;
                    final int stride = workers;
                    futures.add(pool.submit(new Callable<Map<String, float[]>>() {
                        public Map<String, float[]> call() {
                            Map<String, float[]> g = zeros();
                            for (int r = first; r < B; r += stride)
                                losses[r] = rowGrads(x[r], y[r], soft == null ? null : soft[r], alpha == null ? 0 : alpha[r], W, g, N);
                            return g;
                        }
                    }));
                }
                for (Future<Map<String, float[]>> f : futures) parts.add(f.get());
            } finally {
                pool.shutdown();
            }
        }
        Map<String, float[]> g = parts.get(0);
        for (int i = 1; i < parts.size(); i++)
            for (String k : names) {
                float[] a = g.get(k), c = parts.get(i).get(k);
                for (int j = 0; j < a.length; j++) a[j] += c[j];
            }
        double loss = 0;
        for (double l : losses) loss += l;
        return new Object[]{(float) (loss / N), g};
    }

    /** One AdamW step (PyTorch semantics, as grad.py), gradient-norm clipped. Returns the loss. */
    @SuppressWarnings("unchecked")
    public float step(int[][] x, int[][] y, float lr, float[][] soft, float[] alpha) throws Exception {
        Object[] lg = lossAndGrads(x, y, soft, alpha, effective());
        Map<String, float[]> g = (Map<String, float[]>) lg[1];
        if (rank > 0) return adapterStep(g, lr, (Float) lg[0]);
        if (gradClip > 0) {
            double norm = 0;
            for (String k : names) for (float a : g.get(k)) norm += (double) a * a;
            norm = Math.sqrt(norm);
            if (norm > gradClip) {
                float f = (float) (gradClip / (norm + 1e-6));
                for (String k : names) { float[] a = g.get(k); for (int j = 0; j < a.length; j++) a[j] *= f; }
            }
        }
        t++;
        float c1 = (float) (1 - Math.pow(beta1, t)), c2 = (float) (1 - Math.pow(beta2, t));
        for (String k : names) {
            float[] p = w.get(k), gm = g.get(k), mm = m.get(k), vv = v.get(k);
            float decay = 1 - lr * weightDecay;
            for (int j = 0; j < p.length; j++) {
                p[j] *= decay;
                mm[j] = beta1 * mm[j] + (1 - beta1) * gm[j];
                vv[j] = beta2 * vv[j] + (1 - beta2) * gm[j] * gm[j];
                p[j] -= lr * (mm[j] / c1) / ((float) Math.sqrt(vv[j] / c2) + 1e-8f);
            }
        }
        return (Float) lg[0];
    }

    /** Chain rule through W = Q(W0) + B A: gB = gW A^T, gA = B^T gW; AdamW on the adapters only. */
    float adapterStep(Map<String, float[]> gW, float lr, float loss) {
        Map<String, float[]> gA = new HashMap<>(), gB = new HashMap<>();
        double norm = 0;
        for (String k : A.keySet()) {
            int[] sh = base.shape.get(k);
            int rows = sh[0], cols = sh[1];
            float[] g = gW.get(k), a = A.get(k), b = Bm.get(k);
            float[] ga = new float[a.length], gb = new float[b.length];
            for (int o = 0; o < rows; o++)
                for (int r = 0; r < rank; r++) {
                    float sb = 0f, br = b[o * rank + r];
                    for (int i = 0; i < cols; i++) {
                        float gi = g[o * cols + i];
                        sb += gi * a[r * cols + i];
                        ga[r * cols + i] += br * gi;
                    }
                    gb[o * rank + r] = sb;
                }
            for (float f : ga) norm += (double) f * f;
            for (float f : gb) norm += (double) f * f;
            gA.put(k, ga);
            gB.put(k, gb);
        }
        norm = Math.sqrt(norm);
        float clip = (gradClip > 0 && norm > gradClip) ? (float) (gradClip / (norm + 1e-6)) : 1f;
        t++;
        float c1 = (float) (1 - Math.pow(beta1, t)), c2 = (float) (1 - Math.pow(beta2, t));
        for (String k : A.keySet()) {
            adam(A.get(k), gA.get(k), mA.get(k), vA.get(k), lr, clip, c1, c2);
            adam(Bm.get(k), gB.get(k), mB.get(k), vB.get(k), lr, clip, c1, c2);
        }
        return loss;
    }

    void adam(float[] p, float[] g, float[] m, float[] v, float lr, float clip, float c1, float c2) {
        for (int j = 0; j < p.length; j++) {
            float gj = g[j] * clip;
            m[j] = beta1 * m[j] + (1 - beta1) * gj;
            v[j] = beta2 * v[j] + (1 - beta2) * gj * gj;
            p[j] -= lr * (m[j] / c1) / ((float) Math.sqrt(v[j] / c2) + 1e-8f);
        }
    }

    // ------------------------------------------------------------------ saving what was learned

    /** Saves what was learned: just the adapters (plus optimizer state) in adapter mode. */
    public void save(OutputStream os) throws IOException {
        DataOutputStream out = new DataOutputStream(os);
        out.writeUTF("MORPHEUS-LEARNED-2");
        out.writeInt(t);
        out.writeInt(rank);
        List<Map<String, float[]>> parts = rank > 0 ? java.util.Arrays.asList(A, Bm, mA, vA, mB, vB)
                                                    : java.util.Arrays.asList(w, m, v);
        List<String> keys = new ArrayList<>(new java.util.TreeSet<>(rank > 0 ? A.keySet() : w.keySet()));
        out.writeInt(keys.size());
        for (String k : keys) {
            out.writeUTF(k);
            for (Map<String, float[]> mp : parts) {
                float[] a = mp.get(k);
                out.writeInt(a.length);
                for (float f : a) out.writeFloat(f);
            }
        }
        out.flush();
    }

    /** Continue from saved learning: `brain` supplies the config (the brain shipped in the app). */
    public static Trainer load(Brain brain, InputStream is) throws IOException {
        DataInputStream in = new DataInputStream(is);
        String tag = in.readUTF();
        Trainer tr = new Trainer(brain);
        if (tag.equals("MORPHEUS-LEARNED-1")) {
            tr.t = in.readInt();
        } else if (tag.equals("MORPHEUS-LEARNED-2")) {
            tr.t = in.readInt();
            tr.rank = in.readInt();
        } else {
            throw new IOException("not a learned-brain file");
        }
        List<Map<String, float[]>> parts = tr.rank > 0 ? java.util.Arrays.asList(tr.A, tr.Bm, tr.mA, tr.vA, tr.mB, tr.vB)
                                                       : java.util.Arrays.asList(tr.w, tr.m, tr.v);
        int n = in.readInt();
        for (int i = 0; i < n; i++) {
            String k = in.readUTF();
            for (Map<String, float[]> mp : parts) {
                float[] a = new float[in.readInt()];
                for (int j = 0; j < a.length; j++) a[j] = in.readFloat();
                mp.put(k, a);
            }
        }
        return tr;
    }
}
