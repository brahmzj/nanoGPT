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
    public interface World extends Curiosity.Store {
        List<String> readList(String name);
        void writeList(String name, List<String> items);
        boolean internetAllowed();
        String[] lookup(String topic);          // {title, extract}, or null
        String randomArticle();                 // a short article, or null
        String[] article(String topic);         // {title, text, links one per line}, or null
        String fetchText(String url);           // the text of a web page, or null
        List<String> search(String query);      // titles of pages about it, best first (a full-text search)
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
        public float before, after, taughtBefore = -1, taughtAfter = -1, readBefore = -1, readAfter = -1;
        public int steps, newSources, newFacts, wondered;
        public List<String> titles = new ArrayList<>(), foundOut = new ArrayList<>();
        /** For a notification: "question? answer" found, things read as asked, questions it now needs help with. */
        public List<String> found = new ArrayList<>(), reports = new ArrayList<>(), stuck = new ArrayList<>();
        public String summary;
    }

    /** Pages read per session: enough to learn something, little enough for data and battery. */
    public int pagesPerSession = 6;
    /** New facts from reading studied per session: a small brain learns a few at a time, not a page. */
    public int studyPerSession = 4;
    /** Its own questions (from gaps in what it knows): a few per session, after yours. */
    public int ownQuestionsPerSession = 2;

    static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList(
        "what who where when which how is are was were the a an of to in on for do does did you your my me i it and or"
            .split(" ")));

    final List<String> lessons = new ArrayList<>();   // a snapshot of the curriculum, for review
    final List<String[]> exam = new ArrayList<>();    // {stage, prompt, answer}
    public int batch = 8, maxNewRows = 2;
    public float lr = 1e-3f, remember = 0.5f, maxForgetting = 0.02f;
    /** No single skill (a section of the exam) may fall further than this: averages hide a lot. */
    public float maxSkillDrop = 0.075f;
    final Random rng = new Random();

    /** Research only (no studying): for working on your tasks in the background. See {@link #research}. */
    public Learner() {}

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

    /**
     * Browse and absorb (only if the internet is allowed): its curiosity first (the questions it
     * could not answer, most interesting first, then a few of its own), then your interests ("learn
     * about volcanoes") and a few links from each page, then pages you shared. Every page goes into
     * its memory, and its simple sentences become questions and answers to study.
     */
    int gather(World world, Progress progress, Result res) { return gather(world, progress, res, false); }

    /**
     * Work on your tasks, and only those, without studying: your questions, what you asked it to
     * read, and pages you shared. Light work (a few pages), for the background job that keeps going
     * until every task is done. The facts it finds are studied at the next learning session.
     */
    public Result research(World world, Progress progress) {
        Result res = new Result();
        gather(world, progress, res, true);
        return res;
    }

    /** Is any task of yours still waiting (and can it work on it: the internet is allowed)? */
    public static boolean hasWork(World world) {
        if (!world.internetAllowed()) return false;
        for (Curiosity.Wonder w : new Curiosity(world).tasks()) if (w.status.equals("open")) return true;
        for (String q : world.readList("reading_queue")) if (q.endsWith("\t0") || !q.contains("\t")) return true;
        return !world.readList("shared_links").isEmpty();
    }

    /** Where to look, harder each try: the article about it, then a full-text search, then its key words. */
    static List<String> whereToLook(World world, Curiosity.Wonder w) {
        if (w.tries == 0) return Curiosity.topics(w.question);
        List<String> found = world.search(w.tries == 1 ? w.question : Assistant.join(" ", new ArrayList<>(Assistant.keywords(w.question))));
        return found == null ? new ArrayList<String>() : found.subList(0, Math.min(3, found.size()));
    }

    int gather(World world, Progress progress, Result res, boolean background) {
        if (!world.internetAllowed()) return 0;
        int pages = 0;
        List<String> seen = world.readList("read_titles");
        Curiosity curiosity = new Curiosity(world);
        if (!background) wonderAboutGaps(world, curiosity);
        int own = 0;
        for (Curiosity.Wonder w : curiosity.mostInteresting(pagesPerSession, Curiosity.known(world))) {  // 1. curiosity
            if (pages >= pagesPerSession || progress.cancelled()) break;
            if (w.origin.equals("itself") && (background || own++ >= ownQuestionsPerSession)) continue;
            res.wondered++;
            String answer = null, title = null;
            for (String topic : whereToLook(world, w)) {
                if (pages >= pagesPerSession) break;
                progress.update("wondering: " + w.question + "?");
                String[] page = world.article(topic);
                pages++;
                if (page == null) continue;
                absorb(world, page, seen, res);
                answer = answerFrom(page, w.question);
                if (answer != null) { title = page[0]; break; }
            }
            if (answer == null) {  // three misses, and it asks you
                if (curiosity.tried(w.question) && !w.origin.equals("itself")) res.stuck.add(w.question + "?");
                continue;
            }
            if (answer.length() <= 120) teach(world, w.question, answer);
            curiosity.found(w.question, answer, title == null ? "" : "simple wikipedia: " + title.toLowerCase(java.util.Locale.ROOT));
            res.foundOut.add(w.question);
            if (!w.origin.equals("itself")) res.found.add(w.question + "? " + answer);
        }
        List<String> queue = world.readList("reading_queue"), later = new ArrayList<>();  // 2. your interests, and where they lead
        while (pages < pagesPerSession && !queue.isEmpty() && !progress.cancelled()) {
            String line = queue.remove(0);
            String[] entry = line.split("\t");
            int depth = entry.length > 1 ? Integer.parseInt(entry[1]) : 0;
            if (background && depth > 0) { later.add(line); continue; }  // wandering off through links waits for a session
            if (seen.contains(entry[0].toLowerCase(java.util.Locale.ROOT))) {
                if (depth == 0) tell(curiosity, res, "you asked me to learn about " + entry[0] + ". i already read about it: ask me, what do you know about " + entry[0] + "?");
                continue;
            }
            progress.update("reading about " + entry[0] + "\u2026");
            String[] page = world.article(entry[0]);
            pages++;
            if (page == null) {
                if (depth == 0) tell(curiosity, res, "you asked me to learn about " + entry[0] + ", but i could not find anything to read about it.");
                continue;
            }
            int factsBefore = res.newFacts;
            absorb(world, page, seen, res);
            if (depth == 0) {  // a task of yours: done
                List<String> first = Reader.sentences(page[0], page[1]);
                tell(curiosity, res, "you asked me to learn about " + entry[0] + ". i read \"" + page[0].toLowerCase(java.util.Locale.ROOT)
                     + "\"" + (first.isEmpty() ? "" : ": " + first.get(0)) + " i found " + (res.newFacts - factsBefore)
                     + " facts to study the next time you charge me.");
            }
            if (depth < 2 && page.length > 2) {
                int added = 0;
                for (String link : page[2].split("\n")) {
                    String l = link.trim().toLowerCase(java.util.Locale.ROOT);
                    if (l.isEmpty() || l.contains(":") || seen.contains(l) || added >= 3) continue;
                    queue.add(link.trim() + "\t" + (depth + 1));
                    added++;
                }
            }
        }
        later.addAll(queue);
        queue = later;
        while (queue.size() > 40) queue.remove(queue.size() - 1);
        world.writeList("reading_queue", queue);
        List<String> links = world.readList("shared_links");  // 3. pages you shared
        for (String url : links) {
            String text = world.fetchText(url);
            if (text != null && text.length() > 40) {
                absorb(world, new String[]{null, text, ""}, seen, res);
                tell(curiosity, res, "i read the page you shared: " + url);
            }
        }
        world.writeList("shared_links", new ArrayList<String>());
        if (!background && pages == 0 && links.isEmpty() && !progress.cancelled()) {  // 4. nothing asked: a little reading of its own
            String text = world.randomArticle();
            if (text != null) {
                int nl = text.indexOf('\n');
                absorb(world, new String[]{nl > 0 ? text.substring(0, nl) : null, text, ""}, seen, res);
            }
        }
        world.writeList("read_titles", seen.size() > 500 ? seen.subList(seen.size() - 500, seen.size()) : seen);
        return res.newSources;
    }

    static void tell(Curiosity curiosity, Result res, String line) {
        curiosity.report(line);
        res.reports.add(line);
    }

    /**
     * Its own questions: what its knowledge raises (France has a capital and Spain is a country too:
     * so what is the capital of Spain?), skipping what it already knows or already wonders about.
     */
    void wonderAboutGaps(World world, Curiosity curiosity) {
        Set<String> skip = new HashSet<>();
        for (String list : new String[]{"taught", "absorbed"})
            for (String line : world.readList(list)) { String[] f = fact(line); if (f != null) skip.add(f[0]); }
        for (Curiosity.Wonder w : curiosity.all()) skip.add(w.question);
        Reasoner reasoner = new Reasoner(new Reader.Memory(world.library()).sentences);
        for (String q : reasoner.gaps(3, skip)) curiosity.wonder(q, "", "itself");
    }

    /** The answer to a question in a page: a fact it states, else the sentence that covers the question. */
    static String answerFrom(String[] page, String question) {
        String q = Assistant.key(question);
        for (String[] f : Reader.facts(page[0], page[1])) if (f[0].equals(q)) return f[1];
        String hit = new Reader.Memory(Arrays.asList((page[0] != null ? page[0].toLowerCase(java.util.Locale.ROOT) + "\n" : "") + page[1]))
            .recall(question);
        if (hit == null) return null;
        return Reader.shorten(hit) != null ? Reader.shorten(hit) : hit;
    }

    /** Remember a page, and turn its simple sentences into questions and answers to study. */
    void absorb(World world, String[] page, List<String> seen, Result res) {
        String title = page[0], text = page[1];
        if (title != null) {
            seen.add(title.toLowerCase(java.util.Locale.ROOT));
            if (!res.titles.contains(title)) res.titles.add(title);
        }
        world.addToLibrary((title != null ? title + "\n" : "") + text);
        res.newSources++;
        List<String> absorbed = world.readList("absorbed");
        java.util.Set<String> known = new java.util.HashSet<>();
        for (String f : absorbed) known.add(f.substring(0, Math.max(0, f.indexOf('\t'))));
        for (String[] f : Reader.facts(title, text)) {
            if (known.add(f[0])) { absorbed.add(f[0] + "\t" + f[1]); res.newFacts++; }
        }
        while (absorbed.size() > 300) absorbed.remove(0);  // the most recent reading stays in play
        world.writeList("absorbed", absorbed);
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
        int right = 0;
        for (boolean ok : gradeEach(tr, W, questions)) if (ok) right++;
        return (float) right / questions.size();
    }

    /** Right or wrong, per question. */
    static boolean[] gradeEach(final Trainer tr, final Map<String, float[]> W, List<String[]> questions) throws Exception {
        boolean[] out = new boolean[questions.size()];
        if (questions.isEmpty()) return out;
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
            for (int i = 0; i < out.length; i++) out[i] = results.get(i).get();
            return out;
        } finally {
            pool.shutdown();
        }
    }

    /** Each exam section's score (letters, numbers, ..., copying). */
    Map<String, Float> skillScores(Trainer tr, Map<String, float[]> W) throws Exception {
        Map<String, Float> out = new java.util.LinkedHashMap<>();
        List<String> stages = new ArrayList<>();
        for (String[] q : exam) if (!stages.contains(q[0])) stages.add(q[0]);
        for (String st : stages) {
            List<String[]> qs = new ArrayList<>();
            for (String[] q : exam) if (q[0].equals(st)) qs.add(q);
            out.put(st, grade(tr, W, qs));
        }
        return out;
    }

    /** Mean over sections, like the Python report card average. */
    static float average(Map<String, Float> skills) {
        float sum = 0;
        for (float v : skills.values()) sum += v;
        return skills.isEmpty() ? 1f : sum / skills.size();
    }

    // ------------------------------------------------------------------ a session

    /**
     * One learning session on `tr` (changed in place). The caller saves it if result.kept, and
     * otherwise throws it away. Stops early (and keeps nothing) if progress.cancelled().
     */
    public Result session(final Trainer tr, World world, Progress progress, int maxSteps, long budgetMillis) throws Exception {
        long started = System.currentTimeMillis();
        Result res = new Result();
        gather(world, progress, res);

        final List<String[]> facts = new ArrayList<>(), studying = new ArrayList<>(), review = new ArrayList<>();
        for (String line : world.readList("taught")) { String[] f = fact(line); if (f != null) facts.add(f); }
        // don't cram: a few new facts from reading per session (oldest first), plus a little review
        // of facts already mastered (spaced repetition). Raw text only goes to memory, for recall.
        List<String> mastered = world.readList("mastered");
        for (String line : world.readList("absorbed")) {
            String[] f = fact(line);
            if (f == null) continue;
            if (!mastered.contains(f[0])) { if (studying.size() < studyPerSession) studying.add(f); }
            else review.add(f);
        }
        java.util.Collections.shuffle(review, rng);
        while (review.size() > 8) review.remove(review.size() - 1);
        final List<String[]> read = new ArrayList<>(studying);
        read.addAll(review);
        float wTaught = facts.isEmpty() ? 0 : Math.min(1f, 0.2f + facts.size() / 5f);
        float wRead = read.isEmpty() ? 0 : Math.min(1f, 0.2f + read.size() / 10f);
        float amount = Math.min(1f, wTaught + wRead);
        int nNew = amount == 0 ? 0 : Math.min(maxNewRows, Math.max(1, Math.round(maxNewRows * amount)));
        if (facts.isEmpty() && read.isEmpty()) {
            res.summary = "nothing new to learn yet. teach me something, share an article with me, "
                          + "or tap the globe and ask me to learn about something.";
            return res;
        }

        List<String[]> taughtExam = new ArrayList<>(), readExam = new ArrayList<>();
        for (String[] f : facts) taughtExam.add(new String[]{"taught", "user: " + f[0] + "?\nmorpheus: ", f[1] + "\n"});
        for (String[] f : read) readExam.add(new String[]{"read", "user: " + f[0] + "?\nmorpheus: ", f[1] + "\n"});
        final Map<String, float[]> pastSelf = tr.effective();  // learning without forgetting
        progress.update("sitting my exam before studying…");
        Map<String, Float> skillsBefore = skillScores(tr, pastSelf);
        res.before = average(skillsBefore);
        Map<String, Float> skillsFirst = new java.util.HashMap<>();  // each skill's very first score
        for (String line : world.readList("original_skills")) {
            String[] p = line.split("\t");
            skillsFirst.put(p[0], Float.parseFloat(p[1]));
        }
        if (skillsFirst.isEmpty()) {
            List<String> save = new ArrayList<>();
            for (Map.Entry<String, Float> e : skillsBefore.entrySet()) save.add(e.getKey() + "\t" + e.getValue());
            world.writeList("original_skills", save);
            skillsFirst.putAll(skillsBefore);
        }
        res.taughtBefore = grade(tr, pastSelf, taughtExam);
        res.readBefore = grade(tr, pastSelf, readExam);
        String original = first(world.readList("original_exam"));
        // kept only if it lost at most maxForgetting against BOTH its last brain and its very first one
        float first = original == null ? res.before : Float.parseFloat(original);
        float floor = Math.max(res.before, first) - maxForgetting;
        // below its very first score? then this session heals first: fewer new rows, more anchoring
        boolean healing = res.before < first - 0.005f;
        float anchor = healing ? Math.max(remember, 0.8f) : remember;
        if (healing) nNew = Math.min(nNew, 1);
        if (original == null) world.writeList("original_exam", Arrays.asList(String.valueOf(res.before)));

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
                    alpha[r] = anchor;
                } else {
                    List<String> pool = new ArrayList<>();
                    for (int k = 0; k < 16; k++)
                        pool.add(rng.nextFloat() * (wTaught + wRead) < wTaught ? taughtLesson(facts) : taughtLesson(read));
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
        Map<String, Float> skillsAfter = skillScores(tr, now);
        res.after = average(skillsAfter);
        res.taughtAfter = grade(tr, now, taughtExam);
        boolean[] readOk = gradeEach(tr, now, readExam);
        int readRight = 0;
        for (boolean ok : readOk) if (ok) readRight++;
        res.readAfter = readExam.isEmpty() ? -1 : (float) readRight / readExam.size();
        res.kept = res.after >= floor;
        String worse = null;  // and no single skill may fall far below its best (last or first)
        for (Map.Entry<String, Float> e : skillsAfter.entrySet()) {
            float best = Math.max(skillsBefore.containsKey(e.getKey()) ? skillsBefore.get(e.getKey()) : 0f,
                                  skillsFirst.containsKey(e.getKey()) ? skillsFirst.get(e.getKey()) : 0f);
            if (e.getValue() < best - maxSkillDrop) { res.kept = false; worse = e.getKey(); }
        }
        if (res.kept) {  // facts it now answers are mastered; the rest stay on the study list
            for (int i = 0; i < studying.size(); i++) if (readOk[i] && !mastered.contains(studying.get(i)[0])) mastered.add(studying.get(i)[0]);
            world.writeList("mastered", mastered);
        }
        long seconds = (System.currentTimeMillis() - started) / 1000;
        StringBuilder what = new StringBuilder();
        if (res.wondered > 0)
            what.append("i was curious about ").append(res.wondered).append(res.wondered == 1 ? " question" : " questions")
                .append(" and found out ").append(res.foundOut.size()).append(". ");
        if (!res.titles.isEmpty())
            what.append("i read about ").append(Assistant.join(", ", res.titles.subList(0, Math.min(4, res.titles.size()))))
                .append(res.titles.size() > 4 ? " and more" : "").append(" and found ").append(res.newFacts)
                .append(" new facts. ");
        if (res.kept) {
            what.append("i studied");
            if (!facts.isEmpty())
                what.append(String.format(java.util.Locale.US, " %d thing%s you taught me (%.0f%% right)", facts.size(),
                                          facts.size() == 1 ? "" : "s", res.taughtAfter * 100));
            if (!read.isEmpty())
                what.append(facts.isEmpty() ? "" : " and").append(String.format(java.util.Locale.US,
                    " %d facts from my reading (%d right now, %d mastered in all)", read.size(), readRight, mastered.size()));
            what.append(". ");
        } else {
            what.append("i tried to learn, but it made me forget")
                .append(worse != null ? " some " + ("copying".equals(worse) ? "spelling of new words" : worse) : "")
                .append(", so i kept my old brain. ");
        }
        res.summary = what.toString()
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
