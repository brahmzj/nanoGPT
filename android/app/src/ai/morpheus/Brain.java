package ai.morpheus;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/**
 * Morpheus' brain in plain Java: a mirror of morpheus/runtime.py for Android.
 *
 * It reads brain.bin (a .morph file with the LZMA layer removed by export_brain.py, since
 * Android has no LZMA decoder), unpacks the weights (int8, int4, int3, ternary or binary),
 * and thinks one character at a time with a KV cache. No Android classes are used here,
 * so it is tested against the Python runtime on an ordinary JVM.
 */
public final class Brain {

    /** Newline, then printable ASCII: the same 96-symbol alphabet as morpheus/tokenizer.py. */
    public static final String CHARS;
    static {
        StringBuilder sb = new StringBuilder("\n");
        for (char c = 32; c < 127; c++) sb.append(c);
        CHARS = sb.toString();
    }

    final int vocab, block, nLayer, nLoop, nHead, nKv, nEmbd, hd, hidden;
    final float ropeBase;
    final Map<String, float[]> w = new HashMap<>();  // row-major (out, in), like torch
    final float[][] cos, sin;

    public Brain(InputStream in) throws IOException {
        byte[] data = readAll(in);
        String magic = "MORPHRAW\n";
        if (!new String(data, 0, magic.length(), StandardCharsets.US_ASCII).equals(magic))
            throw new IOException("not a brain.bin file");
        int headerEnd = indexOf(data, "\nend\n".getBytes(StandardCharsets.US_ASCII), magic.length() - 1) + 5;
        String header = new String(data, magic.length(), headerEnd - magic.length(), StandardCharsets.US_ASCII);
        Map<String, String> cfg = new HashMap<>();
        for (String line : header.split("\n")) {
            String[] p = line.trim().split(" ");
            if (p.length == 2) cfg.put(p[0], p[1]);
        }
        if (!CHARS.equals(Alphabet.decode(cfg.get("alphabet")))) throw new IOException("different alphabet");
        vocab = Integer.parseInt(cfg.get("vocab_size"));
        block = Integer.parseInt(cfg.get("block_size"));
        nLayer = Integer.parseInt(cfg.get("n_layer"));
        nLoop = Integer.parseInt(cfg.get("n_loop"));
        nHead = Integer.parseInt(cfg.get("n_head"));
        nKv = Integer.parseInt(cfg.get("n_kv_head"));
        nEmbd = Integer.parseInt(cfg.get("n_embd"));
        ropeBase = Float.parseFloat(cfg.get("rope_base"));
        hd = nEmbd / nHead;
        hidden = Integer.parseInt(cfg.get("mlp_ratio")) * nEmbd;
        for (String line : header.split("\n")) {
            String[] p = line.trim().split(" ");
            if (p.length == 8 && p[0].equals("tensor")) {
                String name = p[1], q = p[2];
                int rows = Integer.parseInt(p[3]), cols = Integer.parseInt(p[4]), pad = Integer.parseInt(p[5]);
                int offset = Integer.parseInt(p[6]), size = Integer.parseInt(p[7]);
                w.put(name, dequantize(data, headerEnd + offset, size, q, rows, cols, pad));
            }
        }
        // rotary tables, computed in float32 like numpy
        int half = hd / 2;
        float[] inv = new float[half];
        for (int i = 0; i < half; i++) inv[i] = (float) Math.pow(ropeBase, -((float) i / half));
        cos = new float[block][half];
        sin = new float[block][half];
        for (int t = 0; t < block; t++)
            for (int i = 0; i < half; i++) {
                float a = (float) t * inv[i];
                cos[t][i] = (float) Math.cos(a);
                sin[t][i] = (float) Math.sin(a);
            }
    }

    // ------------------------------------------------------------------ unpacking the weights

    static final int[] POW3 = {81, 27, 9, 3, 1};

    static float[] dequantize(byte[] d, int at, int size, String q, int rows, int cols, int pad) {
        ByteBuffer buf = ByteBuffer.wrap(d, at, size).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[rows * cols];
        if (q.equals("f32")) {
            for (int i = 0; i < out.length; i++) out[i] = buf.getFloat();
            return out;
        }
        if (q.equals("f16")) {
            for (int i = 0; i < out.length; i++) out[i] = half(buf.getShort());
            return out;
        }
        int group;  // weights per scale; 0 means one scale per row
        switch (q) {
            case "int4": case "int3": group = 32; break;
            case "int8": case "ternary": case "binary": group = 0; break;
            default: throw new IllegalArgumentException("unknown scheme " + q);
        }
        int padded = cols + pad, gsize = group == 0 ? padded : group, groups = padded / gsize;
        float[] scale = new float[rows * groups];
        for (int i = 0; i < scale.length; i++) scale[i] = half(buf.getShort());
        int base = at + 2 * rows * groups, n = rows * padded;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                int i = r * padded + c;  // index among the stored (padded) levels
                int level;
                switch (q) {
                    case "int8": level = d[base + i]; break;
                    case "int4": { int b = d[base + i / 2] & 0xFF; level = ((i % 2 == 0) ? (b >> 4) : (b & 0x0F)) - 8; break; }
                    case "int3": level = (d[base + i] & 0xFF) - 4; break;
                    case "ternary": level = ((d[base + i / 5] & 0xFF) / POW3[i % 5]) % 3 - 1; break;
                    default: level = ((d[base + i / 8] >> (7 - i % 8)) & 1) == 1 ? 1 : -1;  // binary, np.packbits order
                }
                out[r * cols + c] = level * scale[r * groups + c / gsize];
            }
        return out;
    }

    static float half(short h) {
        int s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
        float v;
        if (e == 0) v = (float) (m * Math.pow(2, -24));
        else if (e == 31) v = m == 0 ? Float.POSITIVE_INFINITY : Float.NaN;
        else v = Float.intBitsToFloat(((e - 15 + 127) << 23) | (m << 13));
        return s == 1 ? -v : v;
    }

    // ------------------------------------------------------------------ thinking

    /** Keys and values for every (looped) layer, so each new character costs one position of work. */
    public final class Cache {
        final float[][][] k = new float[nLayer * nLoop][block][nKv * hd];
        final float[][][] v = new float[nLayer * nLoop][block][nKv * hd];
        int pos = 0;
    }

    public Cache newCache() { return new Cache(); }

    static void rmsNorm(float[] x, float[] out) {
        double ss = 0;
        for (float a : x) ss += (double) a * a;
        float r = (float) (1.0 / Math.sqrt(ss / x.length + 1e-6));
        for (int i = 0; i < x.length; i++) out[i] = x[i] * r;
    }

    /** out[o] = sum_i W[o][i] * x[i], for W stored row-major as (out, in). */
    static void matvec(float[] W, float[] x, float[] out, int nOut, int nIn) {
        for (int o = 0; o < nOut; o++) {
            float s = 0f;
            int row = o * nIn;
            for (int i = 0; i < nIn; i++) s += W[row + i] * x[i];
            out[o] = s;
        }
    }

    void rope(float[] vec, int off, int t) {
        int half = hd / 2;
        for (int i = 0; i < half; i++) {
            float x1 = vec[off + i], x2 = vec[off + i + half];
            vec[off + i] = x1 * cos[t][i] - x2 * sin[t][i];
            vec[off + i + half] = x1 * sin[t][i] + x2 * cos[t][i];
        }
    }

    /** Feed one character; returns the logits for the next one. */
    public float[] step(Cache cache, int token) {
        int t = cache.pos;
        if (t >= block) throw new IllegalStateException("context is full");
        float[] x = new float[nEmbd], h = new float[nEmbd], q = new float[nHead * hd];
        float[] kv = new float[2 * nKv * hd], y = new float[nEmbd], tmp = new float[nEmbd];
        float[] a = new float[hidden], att = new float[block];
        System.arraycopy(w.get("wte.weight"), token * nEmbd, x, 0, nEmbd);
        int groups = nHead / nKv, layer = 0;
        float scale = (float) (1.0 / Math.sqrt(hd));
        for (int l = 0; l < nLoop; l++)
            for (int b = 0; b < nLayer; b++, layer++) {
                String p = "blocks." + b + ".";
                rmsNorm(x, h);
                matvec(w.get(p + "attn.q.weight"), h, q, nHead * hd, nEmbd);
                matvec(w.get(p + "attn.kv.weight"), h, kv, 2 * nKv * hd, nEmbd);
                for (int hh = 0; hh < nHead; hh++) rope(q, hh * hd, t);
                float[] kt = cache.k[layer][t], vt = cache.v[layer][t];
                System.arraycopy(kv, 0, kt, 0, nKv * hd);
                System.arraycopy(kv, nKv * hd, vt, 0, nKv * hd);
                for (int g = 0; g < nKv; g++) rope(kt, g * hd, t);
                for (int hh = 0; hh < nHead; hh++) {  // head hh reads kv head hh / groups
                    int qo = hh * hd, ko = (hh / groups) * hd;
                    float max = Float.NEGATIVE_INFINITY;
                    for (int s = 0; s <= t; s++) {
                        float dot = 0f;
                        float[] ks = cache.k[layer][s];
                        for (int i = 0; i < hd; i++) dot += q[qo + i] * ks[ko + i];
                        att[s] = dot * scale;
                        if (att[s] > max) max = att[s];
                    }
                    float sum = 0f;
                    for (int s = 0; s <= t; s++) { att[s] = (float) Math.exp(att[s] - max); sum += att[s]; }
                    for (int i = 0; i < hd; i++) {
                        float acc = 0f;
                        for (int s = 0; s <= t; s++) acc += att[s] * cache.v[layer][s][ko + i];
                        y[qo + i] = acc / sum;
                    }
                }
                matvec(w.get(p + "attn.o.weight"), y, tmp, nEmbd, nEmbd);
                for (int i = 0; i < nEmbd; i++) x[i] += tmp[i];
                rmsNorm(x, h);
                matvec(w.get(p + "mlp.fc.weight"), h, a, hidden, nEmbd);
                for (int i = 0; i < hidden; i++) { float r = Math.max(a[i], 0f); a[i] = r * r; }
                matvec(w.get(p + "mlp.proj.weight"), a, tmp, nEmbd, hidden);
                for (int i = 0; i < nEmbd; i++) x[i] += tmp[i];
            }
        cache.pos++;
        rmsNorm(x, h);
        float[] logits = new float[vocab];
        matvec(w.get("wte.weight"), h, logits, vocab, nEmbd);
        return logits;
    }

    public static int[] encode(String s) {
        int[] ids = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            int k = CHARS.indexOf(s.charAt(i));
            ids[i] = k < 0 ? CHARS.indexOf('?') : k;
        }
        return ids;
    }

    /** Greedy continuation of `text` up to (not including) a newline, like NumpyMorpheus.generate. */
    public String generate(String text, int maxNew) {
        int[] ids = encode(Alphabet.normalize(text));
        int start = Math.max(0, ids.length - (block - 1));  // keep as much of the prompt as fits
        Cache cache = newCache();
        float[] logits = null;
        for (int i = start; i < ids.length; i++) logits = step(cache, ids[i]);
        maxNew = Math.min(maxNew, block - (ids.length - start));
        StringBuilder out = new StringBuilder();
        for (int n = 0; n < maxNew && logits != null; n++) {
            int best = 0;
            for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
            char c = CHARS.charAt(best);
            if (c == '\n') break;
            out.append(c);
            if (cache.pos >= block) break;
            logits = step(cache, best);
        }
        return out.toString();
    }

    public int blockSize() { return block; }

    // ------------------------------------------------------------------ helpers

    static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[1 << 16];
        for (int n; (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
        return out.toByteArray();
    }

    static int indexOf(byte[] data, byte[] pat, int from) {
        outer:
        for (int i = from; i <= data.length - pat.length; i++) {
            for (int j = 0; j < pat.length; j++) if (data[i + j] != pat[j]) continue outer;
            return i;
        }
        throw new IllegalArgumentException("header end not found");
    }
}
