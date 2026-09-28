package ai.morpheus;

/**
 * Java twin of morpheus/quant.py: quantize-then-dequantize ("fake quantization"), so a
 * compressed brain keeps learning in its own format. While learning, the forward pass sees
 * these grid values and the updates go to float weights underneath (straight-through).
 */
public final class Quant {

    private Quant() {}

    /** The weights a tensor of `scheme` actually thinks with. `w` is row-major (rows, cols). */
    public static float[] fake(float[] w, int rows, int cols, String scheme) {
        if (scheme.equals("f32")) return w;
        float[] out = new float[w.length];
        if (scheme.equals("f16")) {
            for (int i = 0; i < w.length; i++) out[i] = Brain.half(toHalf(w[i]));
            return out;
        }
        int levels, group;
        String kind;
        switch (scheme) {
            case "int8": kind = "absmax"; levels = 127; group = 0; break;
            case "int4": kind = "absmax"; levels = 7; group = 32; break;
            case "int3": kind = "absmax"; levels = 3; group = 32; break;
            case "ternary": kind = "absmean"; levels = 1; group = 0; break;
            case "binary": kind = "sign"; levels = 1; group = 0; break;
            default: throw new IllegalArgumentException("unknown scheme " + scheme);
        }
        int size = group == 0 ? cols : group;
        int padded = cols + ((size - cols % size) % size);
        int[] q = new int[size];
        for (int r = 0; r < rows; r++)
            for (int g0 = 0; g0 < padded; g0 += size) {
                int base = r * cols;
                float scale;
                if (kind.equals("absmax")) {
                    float max = 0f;
                    for (int c = g0; c < g0 + size && c < cols; c++) max = Math.max(max, Math.abs(w[base + c]));
                    scale = max / levels;
                } else {
                    float sum = 0f;  // padding zeros count in the mean, as in numpy
                    for (int c = g0; c < g0 + size && c < cols; c++) sum += Math.abs(w[base + c]);
                    scale = sum / size;
                }
                if (kind.equals("absmean")) {  // levels first, then the least-squares scale
                    float s0 = Brain.half(toHalf(scale == 0f ? 1f : scale));
                    float used = 0f, fit = 0f;
                    for (int c = g0; c < g0 + size; c++) {
                        float x = c < cols ? w[base + c] : 0f;
                        q[c - g0] = (int) Math.max(-levels, Math.min(levels, Math.rint(x / s0)));
                        used += Math.abs(q[c - g0]);
                        fit += Math.abs(x) * Math.abs(q[c - g0]);
                    }
                    if (used > 0) scale = fit / used;
                }
                float s = Brain.half(toHalf(scale == 0f ? 1f : scale));
                for (int c = g0; c < g0 + size && c < cols; c++) {
                    float x = w[base + c];
                    int level;
                    if (kind.equals("sign")) level = x >= 0 ? 1 : -1;
                    else if (kind.equals("absmax")) level = (int) Math.max(-levels, Math.min(levels, Math.rint(x / s)));
                    else level = q[c - g0];
                    out[base + c] = level * s;
                }
            }
        return out;
    }

    /** float -> IEEE half, rounding to nearest even (numpy's astype(float16)). */
    static short toHalf(float f) {
        int sign = (Float.floatToRawIntBits(f) >>> 16) & 0x8000;
        float a = Math.abs(f);
        if (Float.isNaN(f)) return (short) (sign | 0x7e00);
        if (a >= 65520f) return (short) (sign | 0x7c00);
        if (a < 6.103515625e-05f) {  // subnormal half (below 2^-14)
            int m = (int) Math.rint(a / 5.9604644775390625e-08);  // units of 2^-24
            return (short) (sign | m);
        }
        int e = Math.getExponent(a);
        int m = (int) Math.rint((a / Math.scalb(1.0, e) - 1.0) * 1024);
        if (m == 1024) { e++; m = 0; }
        if (e > 15) return (short) (sign | 0x7c00);
        return (short) (sign | ((e + 15) << 10) | m);
    }
}
