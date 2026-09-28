import ai.morpheus.Assistant;
import ai.morpheus.Brain;

import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
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
        Brain brain;
        try (FileInputStream in = new FileInputStream(args[1])) {
            brain = new Brain(in);
        }
        StringBuilder out = new StringBuilder();
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
            default:
                throw new IllegalArgumentException(args[0]);
        }
        System.out.write(out.toString().getBytes(StandardCharsets.UTF_8));
        System.out.flush();
    }

    static List<String> lines(String path) throws Exception {
        return Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
    }

    static String escape(String s) { return s.replace("\\", "\\\\").replace("\n", "\\n"); }

    static String unescape(String s) { return s.replace("\\n", "\n").replace("\\\\", "\\"); }
}
