import ai.morpheus.Assistant;
import ai.morpheus.Brain;

import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Test driver for the Android brain on a plain JVM (see tests/test_android.py).
 *
 *   logits    <brain.bin> <ids,comma,separated>   logits after feeding the ids one by one
 *   generate  <brain.bin> <prompts.txt>           one reply per prompt ("\n" written as \n)
 *   assistant <brain.bin> <messages.txt> <now>    one assistant reply per message, fixed clock
 */
public class Harness {

    public static void main(String[] args) throws Exception {
        StringBuilder out = new StringBuilder();
        if (args[0].equals("quant")) {  // quant <scheme> <matrix.txt: "rows cols" then the floats>
            List<String> ls = lines(args[2]);
            String[] rc = ls.get(0).trim().split(" ");
            int rows = Integer.parseInt(rc[0]), cols = Integer.parseInt(rc[1]);
            float[] wm = new float[rows * cols];
            for (int i = 0; i < wm.length; i++) wm[i] = Float.parseFloat(ls.get(i + 1));
            for (float f : ai.morpheus.Quant.fake(wm, rows, cols, args[1])) out.append(f).append('\n');
            System.out.write(out.toString().getBytes(StandardCharsets.UTF_8));
            System.out.flush();
            return;
        }
        Brain brain;
        try (FileInputStream in = new FileInputStream(args[1])) {
            brain = new Brain(in);
        }
        switch (args[0]) {
            case "logits": {
                Brain.Cache cache = brain.newCache();
                float[] logits = null;
                for (String id : args[2].split(",")) logits = brain.step(cache, Integer.parseInt(id));
                for (float v : logits) out.append(v).append('\n');
                break;
            }
            case "generate":
                for (String line : lines(args[2])) out.append(escape(brain.generate(unescape(line), 128))).append('\n');
                break;
            case "assistant": {
                final long now = Long.parseLong(args[3]);
                final Map<String, List<String>> store = new HashMap<>();
                Assistant a = new Assistant(brain, new Assistant.Platform() {
                    public long now() { return now; }
                    public List<String> readList(String name) {
                        return new ArrayList<>(store.containsKey(name) ? store.get(name) : new ArrayList<String>());
                    }
                    public void writeList(String name, List<String> items) { store.put(name, new ArrayList<>(items)); }
                    public boolean setAlarm(int hour, int minute, String message) { return false; }
                    public boolean setTimer(int seconds, String message) { return false; }
                    public boolean openUrl(String url) { return false; }
                    public int[] battery() { return null; }
                    public boolean internetAllowed() { return false; }
                    public String[] lookup(String topic) { return null; }
                });
                for (String line : lines(args[2])) out.append(escape(a.respond(line))).append('\n');
                break;
            }
            case "grads": {  // grads <brain.bin> <batch.txt> [<soft.txt> <alpha,alpha,..>]
                ai.morpheus.Trainer tr = new ai.morpheus.Trainer(brain);
                int[][][] xy = batch(args[2]);
                float[][] soft = args.length > 3 ? floats(args[3], xy[0].length) : null;
                float[] alpha = args.length > 4 ? parseFloats(args[4]) : null;
                Object[] lg = tr.lossAndGrads(xy[0], xy[1], soft, alpha, tr.effective());
                out.append(lg[0]).append('\n');
                @SuppressWarnings("unchecked") Map<String, float[]> g = (Map<String, float[]>) lg[1];
                for (String k : new java.util.TreeSet<>(g.keySet())) for (float f : g.get(k)) out.append(f).append('\n');
                break;
            }
            case "train": {  // train <brain.bin> <batch.txt> <steps> <lr>: latent weights after training
                ai.morpheus.Trainer tr = new ai.morpheus.Trainer(brain);
                int[][][] xy = batch(args[2]);
                for (int i = 0; i < Integer.parseInt(args[3]); i++) tr.step(xy[0], xy[1], Float.parseFloat(args[4]), null, null);
                java.io.ByteArrayOutputStream saved = new java.io.ByteArrayOutputStream();
                tr.save(saved);  // and survive a save/load round trip
                tr = ai.morpheus.Trainer.load(brain, new java.io.ByteArrayInputStream(saved.toByteArray()));
                Map<String, float[]> eff = tr.effective();
                for (String k : new java.util.TreeSet<>(eff.keySet())) for (float f : eff.get(k)) out.append(f).append('\n');
                break;
            }
            case "adapt": {  // adapt <brain.bin> <batch.txt> <steps> <lr>: initial A, then A and B after training
                ai.morpheus.Trainer tr = new ai.morpheus.Trainer(brain);
                tr.useAdapters(2, 7);
                int[][][] xy = batch(args[2]);
                java.lang.reflect.Field fa = ai.morpheus.Trainer.class.getDeclaredField("A");
                java.lang.reflect.Field fb = ai.morpheus.Trainer.class.getDeclaredField("Bm");
                fa.setAccessible(true);
                fb.setAccessible(true);
                @SuppressWarnings("unchecked") Map<String, float[]> a0 = (Map<String, float[]>) fa.get(tr);
                for (String k : new java.util.TreeSet<>(a0.keySet())) for (float f : a0.get(k)) out.append(f).append('\n');
                for (int i = 0; i < Integer.parseInt(args[3]); i++) tr.step(xy[0], xy[1], Float.parseFloat(args[4]), null, null);
                java.io.ByteArrayOutputStream saved = new java.io.ByteArrayOutputStream();
                tr.save(saved);  // adapters survive a save/load round trip
                tr = ai.morpheus.Trainer.load(brain, new java.io.ByteArrayInputStream(saved.toByteArray()));
                @SuppressWarnings("unchecked") Map<String, float[]> a1 = (Map<String, float[]>) fa.get(tr);
                @SuppressWarnings("unchecked") Map<String, float[]> b1 = (Map<String, float[]>) fb.get(tr);
                for (String k : new java.util.TreeSet<>(a1.keySet())) {
                    for (float f : a1.get(k)) out.append(f).append('\n');
                    for (float f : b1.get(k)) out.append(f).append('\n');
                }
                break;
            }
            case "session": {  // session <brain.bin> <lessons.txt> <exam.txt> <steps> <question...>
                final Map<String, List<String>> store = new HashMap<>();
                final List<String> lib = new ArrayList<>();
                ai.morpheus.Learner.World world = new ai.morpheus.Learner.World() {
                    public List<String> readList(String name) {
                        return new ArrayList<>(store.containsKey(name) ? store.get(name) : new ArrayList<String>());
                    }
                    public void writeList(String name, List<String> items) { store.put(name, new ArrayList<>(items)); }
                    public boolean internetAllowed() { return false; }
                    public String[] lookup(String topic) { return null; }
                    public String randomArticle() { return null; }
                    public String fetchText(String url) { return null; }
                    public List<String> library() { return new ArrayList<>(lib); }
                    public void addToLibrary(String text) { lib.add(0, text); }
                    public long now() { return 0; }
                };
                store.put("taught", new ArrayList<>(java.util.Arrays.asList(
                    "what is the capital of france\tparis is the capital of france.")));
                lib.add("the moon goes around the earth. the earth goes around the sun.");
                ai.morpheus.Learner learner = new ai.morpheus.Learner(new FileInputStream(args[2]), new FileInputStream(args[3]));
                if (System.getProperty("remember") != null) learner.remember = Float.parseFloat(System.getProperty("remember"));
                if (System.getProperty("lr") != null) learner.lr = Float.parseFloat(System.getProperty("lr"));
                ai.morpheus.Trainer tr = new ai.morpheus.Trainer(brain);
                if (System.getProperty("rank") != null) tr.useAdapters(Integer.parseInt(System.getProperty("rank")), 1);
                ai.morpheus.Learner.Result r = learner.session(tr, world, new ai.morpheus.Learner.Progress() {
                    public void update(String message) { System.err.println(message); }
                    public boolean cancelled() { return false; }
                }, Integer.parseInt(args[4]), 3_600_000L);
                out.append(r.kept).append('\n').append(r.before).append('\n').append(r.after).append('\n')
                   .append(r.taughtBefore).append('\n').append(r.taughtAfter).append('\n').append(r.summary).append('\n');
                Brain learned = tr.toBrain();
                for (int i = 5; i < args.length; i++)
                    out.append(learned.generate("\nuser: " + args[i] + "\nmorpheus: ", 128)).append('\n');
                break;
            }
            default:
                throw new IllegalArgumentException(args[0]);
        }
        System.out.write(out.toString().getBytes(StandardCharsets.UTF_8));
        System.out.flush();
    }

    /** batch.txt: one row per line, "x ids ; y ids" */
    static int[][][] batch(String path) throws Exception {
        List<String> rows = lines(path);
        int[][] x = new int[rows.size()][], y = new int[rows.size()][];
        for (int r = 0; r < rows.size(); r++) {
            String[] xy = rows.get(r).split(";");
            x[r] = ints(xy[0]);
            y[r] = ints(xy[1]);
        }
        return new int[][][]{x, y};
    }

    static int[] ints(String s) {
        String[] p = s.trim().split(" ");
        int[] out = new int[p.length];
        for (int i = 0; i < p.length; i++) out[i] = Integer.parseInt(p[i]);
        return out;
    }

    static float[] parseFloats(String s) {
        String[] p = s.split(",");
        float[] out = new float[p.length];
        for (int i = 0; i < p.length; i++) out[i] = Float.parseFloat(p[i]);
        return out;
    }

    /** one row of floats per line */
    static float[][] floats(String path, int rows) throws Exception {
        List<String> lines = lines(path);
        float[][] out = new float[rows][];
        for (int r = 0; r < rows; r++) {
            String[] p = lines.get(r).trim().split(" ");
            out[r] = new float[p.length];
            for (int i = 0; i < p.length; i++) out[r][i] = Float.parseFloat(p[i]);
        }
        return out;
    }

    static List<String> lines(String path) throws Exception {
        return Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
    }

    static String escape(String s) { return s.replace("\\", "\\\\").replace("\n", "\\n"); }

    static String unescape(String s) { return s.replace("\\n", "\n").replace("\\\\", "\\"); }
}
