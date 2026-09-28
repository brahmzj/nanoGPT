package ai.morpheus;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Self-learning on the phone: a port of the learning session in morpheus/learn.py.
 *
 * Morpheus learns from what you teach it, from its own mistakes (calculator corrections), from
 * its own curiosity (questions it could not answer, looked up when the internet is allowed) and
 * from what you share with it. Every batch mixes review rows (the ABC curriculum, anchored to
 * its own pre-session answers so nothing is forgotten) with new rows (the new material). An
 * exam before and after decides: the new brain is kept only if it did not forget.
 */
public final class Learner {

    /** What learning needs from the device. */
    public interface World {
        List<String> readList(String name);
        void writeList(String name, List<String> items);
        boolean internetAllowed();
        String[] lookup(String topic);          // {title, extract}, or null
        String randomArticle();                 // a short article, or null
        String fetchText(String url);           // the text of a web page, or null
        List<String> library();                 // everything read so far, newest first
        void addToLibrary(String text);
        long now();
    }

    public interface Progress {
        void update(String message);
        boolean cancelled();
    }

    public static final class Result {
        public boolean kept;
        public float before, after, taughtBefore = -1, taughtAfter = -1;
        public int steps, newSources;
        public String summary;
    }

    static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList(
        "what who where when which how is are was were the a an of to in on for do does did you your my me i it and or"
            .split(" ")));

    final List<String> lessons = new ArrayList<>();   // a snapshot of the curriculum, for review
    final List<String[]> exam = new ArrayList<>();    // {stage, prompt, answer}
    public int batch = 8, maxNewRows = 2;
    public float lr = 1e-3f, remember = 0.5f, maxForgetting = 0.02f;
    final Random rng = new Random();

    /** lessons: one per line, '\t' inside a lesson is a newline. exam: stage \t prompt \t answer, "\n" escaped. */
    public Learner(InputStream lessonsIn, InputStream examIn) throws IOException {
        for (String line : read(lessonsIn)) if (!line.isEmpty()) lessons.add(line.replace('\t', '\n'));
        for (String line : read(examIn)) {
            String[] p = line.split("\t");
            if (p.length == 3) exam.add(new String[]{p[0], p[1].replace("\\n", "\n"), p[2].replace("\\n", "\n")});
        }
    }

    static List<String> read(InputStream in) throws IOException {
        List<String> out = new ArrayList<>();
        BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        for (String line; (line = r.readLine()) != null; ) out.add(line);
        return out;
    }

    // ------------------------------------------------------------------ gathering new material

    static final Pattern TOPIC = Pattern.compile(
        "^(what|who) (is|are|was|were) (a |an |the )?(.+?)\\??$|^(tell me about|look up|lookup) (.+?)\\??$");

    static String topicOf(String question) {
        Matcher m = TOPIC.matcher(question);
        if (!m.matches()) return null;
        return m.group(4) != null ? m.group(4) : m.group(6);
    }

    int gather(World world, Progress progress) {
        int found = 0;
        if (!world.internetAllowed()) return 0;
        List<String> wonders = world.readList("wonders"), still = new ArrayList<>();
        int looked = 0;
        for (String q : wonders) {  // its own curiosity: questions it could not answer
            String topic = topicOf(q);
            if (topic == null || looked >= 5 || progress.cancelled()) { if (topic != null) still.add(q); continue; }
            looked++;
            progress.update("looking up " + topic + "…");
            String[] hit = world.lookup(topic);
            if (hit == null) { still.add(q); continue; }
            String answer = Assistant.firstSentences(hit[1], 220).toLowerCase(java.util.Locale.ROOT);
            if (answer.length() <= 100) teach(world, q, Alphabet.normalize(answer));
            world.addToLibrary(hit[0] + "\n" + hit[1]);
            found++;
        }
        world.writeList("wonders", still);
        List<String> links = world.readList("shared_links");  // pages you shared
        for (String url : links) {
            String text = world.fetchText(url);
            if (text != null && text.length() > 40) { world.addToLibrary(text); found++; }
        }
        world.writeList("shared_links", new ArrayList<String>());
        for (int i = 0; i < 3 && !progress.cancelled(); i++) {  // a little reading of its own
            String text = world.randomArticle();
            if (text != null) { world.addToLibrary(text); found++; }
        }
        return found;
    }

    static void teach(World world, String question, String answer) {
        String q = Assistant.key(question);
        List<String> facts = world.readList("taught"), keep = new ArrayList<>();
        for (String f : facts) if (!f.startsWith(q + "\t")) keep.add(f);
        keep.add(q + "\t" + answer.trim());
        world.writeList("taught", keep);
    }

    // ------------------------------------------------------------------ lessons

    static String[] fact(String line) {
        int tab = line.indexOf('\t');
        return tab > 0 ? new String[]{line.substring(0, tab), line.substring(tab + 1)} : null;
    }

    String qa(String q, String a) {
        return "user: " + q + (rng.nextFloat() < 0.8f ? "?" : "") + "\nmorpheus: " + a;
    }

    String pseudoWord() {
        String consonants = "bcdfghjklmnprstvwz", vowels = "aeiou";
        int n = 3 + rng.nextInt(5);
        boolean first = rng.nextBoolean();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            String from = ((i % 2 == 0) == first) ? vowels : consonants;
            sb.append(from.charAt(rng.nextInt(from.length())));
        }
        return sb.toString();
    }

    /** A taught fact, or (a third of the time) a near-miss: the same question about something else. */
    String taughtLesson(List<String[]> facts) {
        String[] f = facts.get(rng.nextInt(facts.size()));
        if (rng.nextInt(3) > 0) return qa(f[0], f[1]);
        String[] words = f[0].split(" ");
        List<Integer> spots = new ArrayList<>();
        for (int i = 0; i < words.length; i++) if (words[i].length() > 2 && !STOPWORDS.contains(words[i])) spots.add(i);
        if (spots.isEmpty()) return qa(f[0], f[1]);
        words[spots.get(rng.nextInt(spots.size()))] = pseudoWord();
        return qa(Assistant.join(" ", Arrays.asList(words)), Assistant.UNKNOWN_ANSWER);
    }

    String readingLesson(String text) {
        int n = 64 + rng.nextInt(193);
        int i = rng.nextInt(Math.max(1, text.length() - n));
        return text.substring(i, Math.min(text.length(), i + n));
    }

    /** A row of block+1 characters: a newline, then lessons separated by newlines (as in school.py). */
    int[] row(int block, List<String> pool) {
        StringBuilder sb = new StringBuilder("\n");
        while (sb.length() < block + 1) sb.append(pool.get(rng.nextInt(pool.size()))).append('\n');
        return Brain.encode(sb.substring(0, block + 1));
    }

    // ------------------------------------------------------------------ exams

    /** Fraction right: greedy decoding says the answer iff every answer position's argmax is right. */
    static float grade(final Trainer tr, final Map<String, float[]> W, List<String[]> questions) throws Exception {
        if (questions.isEmpty()) return -1;
        final int block = tr.base.block;
        ExecutorService pool = Executors.newFixedThreadPool(tr.threads);
        try {
            List<Future<Boolean>> results = new ArrayList<>();
            for (final String[] q : questions)
                results.add(pool.submit(new Callable<Boolean>() {
                    public Boolean call() {
                        int[] all = Brain.encode("\n" + q[1] + q[2]);
                        int start = Math.max(0, all.length - (block + 1));
                        int[] seq = Arrays.copyOfRange(all, start, all.length);
                        int a = seq.length - q[2].length(), V = tr.base.vocab;
                        float[] logits = tr.logits(Arrays.copyOf(seq, seq.length - 1), W);
                        for (int t = a - 1; t < seq.length - 1; t++) {
                            int best = 0;
                            for (int i = 1; i < V; i++) if (logits[t * V + i] > logits[t * V + best]) best = i;
                            if (best != seq[t + 1]) return false;
                        }
                        return true;
                    }
                }));
            int right = 0;
            for (Future<Boolean> f : results) if (f.get()) right++;
            return (float) right / questions.size();
        } finally {
            pool.shutdown();
        }
    }

    /** Mean over stages of each stage's score, like the Python report card average. */
    float curriculumScore(Trainer tr, Map<String, float[]> W) throws Exception {
        List<String> stages = new ArrayList<>();
        for (String[] q : exam) if (!stages.contains(q[0])) stages.add(q[0]);
        float sum = 0;
        for (String st : stages) {
            List<String[]> qs = new ArrayList<>();
            for (String[] q : exam) if (q[0].equals(st)) qs.add(q);
            sum += grade(tr, W, qs);
        }
        return stages.isEmpty() ? 1f : sum / stages.size();
    }

    // ------------------------------------------------------------------ a session

    /**
     * One learning session on `tr` (changed in place). The caller saves it if result.kept, and
     * otherwise throws it away. Stops early (and keeps nothing) if progress.cancelled().
     */
    public Result session(final Trainer tr, World world, Progress progress, int maxSteps, long budgetMillis) throws Exception {
        long started = System.currentTimeMillis();
        Result res = new Result();
        res.newSources = gather(world, progress);

        final List<String[]> facts = new ArrayList<>();
        for (String line : world.readList("taught")) { String[] f = fact(line); if (f != null) facts.add(f); }
        List<String> reading = world.library();
        int readingChars = 0;
        for (String r : reading) readingChars += r.length();
        float amount = Math.min(1f, Math.min(1f, readingChars / 20000f) + (facts.isEmpty() ? 0f : Math.min(1f, 0.2f + facts.size() / 5f)));
        int nNew = (facts.isEmpty() && reading.isEmpty()) ? 0 : Math.min(maxNewRows, Math.max(1, Math.round(maxNewRows * amount)));
        if (nNew == 0 && facts.isEmpty()) {
            res.summary = "nothing new to learn yet. teach me something, share an article with me, or allow the internet.";
            return res;
        }

        List<String[]> taughtExam = new ArrayList<>();
        for (String[] f : facts) taughtExam.add(new String[]{"taught", "user: " + f[0] + "?\nmorpheus: ", f[1] + "\n"});
        final Map<String, float[]> pastSelf = tr.effective();  // learning without forgetting
        progress.update("sitting my exam before studying…");
        res.before = curriculumScore(tr, pastSelf);
        res.taughtBefore = grade(tr, pastSelf, taughtExam);
        String original = first(world.readList("original_exam"));
        float floor = Math.min(res.before, original == null ? res.before : Float.parseFloat(original)) - maxForgetting;
        if (original == null) world.writeList("original_exam", Arrays.asList(String.valueOf(res.before)));

        // weight reading and taught facts by how much there is of each
        float wRead = reading.isEmpty() ? 0 : Math.min(1f, readingChars / 20000f);
        float wTaught = facts.isEmpty() ? 0 : Math.min(1f, 0.2f + facts.size() / 5f);
        final int block = tr.base.block;
        long stepMillis = 0;
        int steps = maxSteps;
        for (int i = 0; i < steps; i++) {
            if (progress.cancelled()) { res.summary = "stopped early; kept my old brain."; return res; }
            long t0 = System.currentTimeMillis();
            int[][] x = new int[batch][], y = new int[batch][];
            final float[][] soft = new float[batch][];
            float[] alpha = new float[batch];
            for (int r = 0; r < batch; r++) {
                int[] rowIds;
                if (r < batch - nNew) {
                    rowIds = row(block, lessons);
                    alpha[r] = remember;
                } else {
                    List<String> pool = new ArrayList<>();
                    for (int k = 0; k < 16; k++)
                        pool.add(rng.nextFloat() * (wRead + wTaught) < wTaught ? taughtLesson(facts)
                                 : readingLesson(reading.get(rng.nextInt(reading.size()))));
                    rowIds = row(block, pool);
                }
                x[r] = Arrays.copyOf(rowIds, block);
                y[r] = Arrays.copyOfRange(rowIds, 1, block + 1);
            }
            ExecutorService pool = Executors.newFixedThreadPool(tr.threads);  // the past self's answers
            try {
                List<Future<float[]>> fs = new ArrayList<>();
                for (int r = 0; r < batch; r++) {
                    if (alpha[r] == 0) { fs.add(null); continue; }
                    final int[] xr = x[r];
                    fs.add(pool.submit(new Callable<float[]>() {
                        public float[] call() {
                            float[] z = tr.logits(xr, pastSelf);
                            Trainer.softmaxRows(z, xr.length, tr.base.vocab);
                            return z;
                        }
                    }));
                }
                for (int r = 0; r < batch; r++) if (fs.get(r) != null) soft[r] = fs.get(r).get();
            } finally {
                pool.shutdown();
            }
            float warm = Math.min(1f, (i + 1) / 10f);
            float lrNow = lr * warm * 0.5f * (1f + (float) Math.cos(Math.PI * i / steps));  // ends in a dream
            float loss = tr.step(x, y, lrNow, soft, alpha);
            if (i == 0) {  // fit the session to the time budget
                stepMillis = System.currentTimeMillis() - t0;
                long left = budgetMillis - (System.currentTimeMillis() - started);
                int fit = (int) (left * 0.8 / Math.max(1, stepMillis));
                steps = Math.min(maxSteps, Math.max(Math.min(10, maxSteps), fit));  // at least 10, at most maxSteps
            }
            if ((i + 1) % 10 == 0 || i == 0)
                progress.update("studying: step " + (i + 1) + "/" + steps + String.format(java.util.Locale.US, ", loss %.3f", loss));
        }
        res.steps = steps;
        progress.update("sitting my exam after studying…");
        Map<String, float[]> now = tr.effective();
        res.after = curriculumScore(tr, now);
        res.taughtAfter = grade(tr, now, taughtExam);
        res.kept = res.after >= floor;
        long seconds = (System.currentTimeMillis() - started) / 1000;
        res.summary = (res.kept ? "i learned " : "i tried to learn, but it made me forget, so i kept my old brain. ")
            + (res.kept ? (facts.isEmpty() ? "from my reading" : facts.size() + " thing" + (facts.size() == 1 ? "" : "s") + " you taught me"
                           + (res.taughtAfter >= 0 ? String.format(java.util.Locale.US, " (%.0f%% right now)", res.taughtAfter * 100) : ""))
                        + (res.newSources > 0 ? " and " + res.newSources + " new thing" + (res.newSources == 1 ? "" : "s") + " i read" : "")
                        + ". " : "")
            + String.format(java.util.Locale.US, "my exam: %.1f%% before, %.1f%% after. %d steps in %d s.",
                            res.before * 100, res.after * 100, steps, seconds);
        List<String> sessions = world.readList("sessions");
        sessions.add(res.summary);
        while (sessions.size() > 20) sessions.remove(0);
        world.writeList("sessions", sessions);
        return res;
    }

    static String first(List<String> list) { return list.isEmpty() ? null : list.get(0); }
}
