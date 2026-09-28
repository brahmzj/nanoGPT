package ai.morpheus;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Morpheus' curiosity: the questions it wants answered, how interesting each one is, and what it
 * found out to tell you.
 *
 * The most basic version of what brains do (information-gap curiosity, Loewenstein; learning
 * progress, Oudeyer): a gap is most interesting when you know something close to it, when it
 * keeps coming up, or when you have a half-formed guess to check; it gets less interesting each
 * time a search fails, and after three failures Morpheus stops searching and asks you instead.
 * "I don't know yet" is never the end: every gap becomes a question it goes after.
 */
public final class Curiosity {

    /** Where the list lives: both the assistant and the learning sessions can reach it. */
    public interface Store {
        List<String> readList(String name);
        void writeList(String name, List<String> items);
        long now();
    }

    public static final class Wonder {
        public String question, status = "open", origin = "you", answer = "", source = "", guess = "";
        public int asked = 1, tries = 0, askedYou = 0;
        public long born, lastTry, lastAskedYou;

        String line() {
            return Assistant.join("\t", java.util.Arrays.asList(question, status, origin, String.valueOf(asked),
                String.valueOf(tries), String.valueOf(askedYou), String.valueOf(born), String.valueOf(lastTry),
                String.valueOf(lastAskedYou), clean(answer), clean(source), clean(guess)));
        }

        static String clean(String s) { return s == null ? "" : s.replace('\t', ' ').replace('\n', ' '); }

        static Wonder parse(String line) {
            String[] p = line.split("\t", -1);
            Wonder w = new Wonder();
            w.question = p[0];
            if (p.length >= 12) {
                w.status = p[1]; w.origin = p[2];
                w.asked = Integer.parseInt(p[3]); w.tries = Integer.parseInt(p[4]); w.askedYou = Integer.parseInt(p[5]);
                w.born = Long.parseLong(p[6]); w.lastTry = Long.parseLong(p[7]); w.lastAskedYou = Long.parseLong(p[8]);
                w.answer = p[9]; w.source = p[10]; w.guess = p[11];
            }
            return w;
        }
    }

    static final long DAY = 24L * 3600 * 1000;
    final Store store;

    public Curiosity(Store store) { this.store = store; }

    public List<Wonder> all() {
        List<Wonder> out = new ArrayList<>();
        for (String line : store.readList("curiosity")) if (!line.isEmpty()) out.add(Wonder.parse(line));
        List<String> legacy = store.readList("wonders");  // questions saved by version 1.1/1.2
        if (!legacy.isEmpty()) {
            for (String q : legacy) if (find(out, q) == null) { Wonder w = new Wonder(); w.question = q; w.born = store.now(); out.add(w); }
            store.writeList("wonders", new ArrayList<String>());
            save(out);
        }
        return out;
    }

    void save(List<Wonder> ws) {
        while (ws.size() > 80) {  // forget the oldest closed questions first
            int drop = 0;
            for (int i = 0; i < ws.size(); i++) if (!ws.get(i).status.equals("open") && !ws.get(i).status.equals("stuck")) { drop = i; break; }
            ws.remove(drop);
        }
        List<String> lines = new ArrayList<>();
        for (Wonder w : ws) lines.add(w.line());
        store.writeList("curiosity", lines);
    }

    static Wonder find(List<Wonder> ws, String question) {
        String q = Assistant.key(question);
        for (Wonder w : ws) if (w.question.equals(q)) return w;
        return null;
    }

    /** A gap: something it could not answer (origin "you"), was unsure of ("unsure") or noticed itself ("itself"). */
    public Wonder wonder(String question, String guess, String origin) {
        List<Wonder> ws = all();
        Wonder w = find(ws, question);
        if (w == null) {
            w = new Wonder();
            w.question = Assistant.key(question);
            w.origin = origin;
            w.born = store.now();
            w.asked = origin.equals("itself") ? 0 : 1;
            ws.add(w);
        } else {
            if (!origin.equals("itself")) w.asked++;
            if (w.status.equals("told") || w.status.equals("found")) return w;  // already known
        }
        if (guess != null && !guess.isEmpty()) w.guess = guess;
        save(ws);
        return w;
    }

    /** How much it wants to know this, right now. */
    public double interest(Wonder w, Set<String> known) {
        if (!w.status.equals("open")) return -1;
        Set<String> k = Assistant.keywords(w.question);
        k.retainAll(known);
        return Math.log(1 + w.asked)                          // it keeps coming up
            + (k.isEmpty() ? 0 : 1.0)                          // it knows something close to it
            + (w.guess.isEmpty() ? 0 : 0.5)                    // a half-formed guess wants checking
            + (store.now() - w.born < DAY ? 0.5 : 0)           // new questions are exciting
            - 0.8 * w.tries;                                   // each failed search makes it less promising
    }

    public List<Wonder> mostInteresting(int n, final Set<String> known) {
        List<Wonder> open = new ArrayList<>();
        for (Wonder w : all()) if (w.status.equals("open")) open.add(w);
        Collections.sort(open, new Comparator<Wonder>() {
            public int compare(Wonder a, Wonder b) { return Double.compare(interest(b, known), interest(a, known)); }
        });
        return open.subList(0, Math.min(n, open.size()));
    }

    /**
     * A search that found nothing. Each try looks harder (the article, then a full-text search, then
     * a search for its key words); after three it stops searching and asks you. True if it is stuck now.
     */
    public boolean tried(String question) {
        List<Wonder> ws = all();
        Wonder w = find(ws, question);
        if (w == null) return false;
        w.tries++;
        w.lastTry = store.now();
        if (w.tries >= TRIES) w.status = "stuck";
        save(ws);
        return w.status.equals("stuck");
    }

    public static final int TRIES = 3;

    /** You called it off ("never mind ..."): the questions that share most of these words. How many. */
    public int drop(String about) {
        Set<String> k = Assistant.keywords(about);
        if (k.isEmpty()) return 0;
        List<Wonder> ws = all();
        int n = 0;
        for (Wonder w : ws) {
            if (!w.status.equals("open") && !w.status.equals("stuck")) continue;
            Set<String> q = Assistant.keywords(w.question);
            q.retainAll(k);
            if (q.size() * 2 >= k.size() + (k.size() > 1 ? 1 : 0)) { w.status = "dropped"; n++; }
        }
        if (n > 0) save(ws);
        return n;
    }

    /** Your questions it is still working on (not its own). */
    public List<Wonder> tasks() {
        List<Wonder> out = new ArrayList<>();
        for (Wonder w : all()) if (!w.origin.equals("itself") && (w.status.equals("open") || w.status.equals("stuck"))) out.add(w);
        return out;
    }

    /** News that is not an answer to a question ("i read about volcanoes, as you asked"): told once. */
    public void report(String line) {
        List<String> r = store.readList("reports");
        r.add(line);
        while (r.size() > 20) r.remove(0);
        store.writeList("reports", r);
    }

    /** It found the answer by itself: news for you, the next time you talk (see {@link #news}). */
    public void found(String question, String answer, String source) { settle(question, answer, source, "found"); }

    /** You told it, or it told you right away: nothing left to report. */
    public void answered(String question, String answer, String source) { settle(question, answer, source, "told"); }

    void settle(String question, String answer, String source, String status) {
        List<Wonder> ws = all();
        Wonder w = find(ws, question);
        if (w == null) {
            if (source.equals("you told me")) return;  // a lesson, not a question it had
            w = new Wonder();
            w.question = Assistant.key(question);
            w.born = store.now();
            ws.add(w);
        }
        w.status = status;
        w.answer = answer;
        w.source = source;
        save(ws);
    }

    /**
     * What it found out about your questions, told once. If it had a guess, it says whether the guess
     * was right: a prediction checked against the world is how a mind learns what to trust.
     */
    public List<String> news() {
        List<Wonder> ws = all();
        List<String> out = new ArrayList<>(store.readList("reports"));
        if (!out.isEmpty()) store.writeList("reports", new ArrayList<String>());
        for (Wonder w : ws)
            if (w.status.equals("found") && !w.origin.equals("itself")) {
                String src = w.source.isEmpty() ? "" : " (" + w.source + ")";
                if (w.guess.isEmpty()) out.add("you asked me \"" + w.question + "?\" i found out: " + w.answer + src);
                else out.add("you asked me \"" + w.question + "?\" i thought \"" + w.guess + "\" and " + (agrees(w.guess, w.answer, w.question)
                             ? "i was right: " : "i was wrong. i found out: ") + w.answer + src);
                w.status = "told";
            }
        if (!out.isEmpty()) save(ws);
        return out;
    }

    /** Does the answer back the guess? Most of what the guess says (beyond the question's own words) must be in it. */
    static boolean agrees(String guess, String answer, String question) {
        Set<String> said = Assistant.keywords(guess), support = Assistant.keywords(answer);
        said.removeAll(Assistant.keywords(question));
        if (said.isEmpty()) return false;
        int backed = 0;
        for (String w : said) if (support.contains(w)) backed++;
        return backed * 2 > said.size();
    }

    /**
     * A question to ask you (children learn most by asking): one it could not find out by itself
     * (or cannot look up, without the internet), stuck ones first, each at most every 12 hours.
     */
    public Wonder toAsk(boolean canSearch) {
        List<Wonder> ws = all();
        Wonder pick = null;
        for (Wonder w : ws) {
            boolean candidate = w.status.equals("stuck") || (w.status.equals("open") && (w.tries >= 1 || !canSearch));
            if (candidate && w.askedYou < 3 && store.now() - w.lastAskedYou > DAY / 2 && (pick == null || w.status.equals("stuck")))
                pick = w;
        }
        if (pick != null) {
            pick.askedYou++;
            pick.lastAskedYou = store.now();
            save(ws);
        }
        return pick;
    }

    public int open() {
        int n = 0;
        for (Wonder w : all()) if (w.status.equals("open") || w.status.equals("stuck")) n++;
        return n;
    }

    /** The words of everything it knows (taught and read questions): what makes a gap feel close. */
    public static Set<String> known(Store store) {
        Set<String> out = new HashSet<>();
        for (String list : new String[]{"taught", "absorbed"})
            for (String line : store.readList(list)) {
                int tab = line.indexOf('\t');
                if (tab > 0) out.addAll(Assistant.keywords(line.substring(0, tab)));
            }
        return out;
    }

    // ------------------------------------------------------------------ where to look

    static final Pattern AFTER_VERB = Pattern.compile(
        "^(?:what|who|whom|whose|where|when|why|how|which)(?: [a-z']+){0,2}? (?:is|are|was|were|do|does|did|can|has|have|had) (.+)$");
    static final Pattern TRAILING_VERB = Pattern.compile(
        " (built|made|born|founded|invented|written|discovered|created|started|named|called|found|have|has|live|eat|mean|do|go|come"
        + "|work|die|begin|end|happen|start|look like|weigh)$");
    static final Pattern WHO_DID = Pattern.compile(
        "^(?:who|what) (?:[a-z]+ed|wrote|made|built|invented|discovered|founded|created|painted|sang|won|found|began|started)"
        + " (?:the |a |an )?([a-z][a-z' -]+)$");
    static final Pattern YES_NO_SUBJECT = Pattern.compile(  // "is the eiffel tower in france" -> "eiffel tower"
        "^(?:is|are|was|were|does|do|did|can|has|have) (?:the |a |an )?([a-z][a-z' -]+?)"
        + " (?:in|on|at|near|from|part of|a|an|the|bigger|smaller|older|taller|longer|made|have|has|live|eat)\\b");
    static final Pattern OF_TAIL = Pattern.compile(" of (?:the |a |an )?([a-z][a-z' -]+)$");

    /** What to read about to answer a question, best first: "how tall is the eiffel tower" -> "eiffel tower". */
    public static List<String> topics(String question) {
        String q = Assistant.key(question);
        Set<String> out = new LinkedHashSet<>();
        String topic = Learner.topicOf(q);
        Matcher who = WHO_DID.matcher(q), yesNo = YES_NO_SUBJECT.matcher(q);
        if (topic == null && who.matches()) topic = who.group(1);
        if (topic == null && yesNo.lookingAt()) topic = yesNo.group(1);
        if (topic == null) {
            Matcher m = AFTER_VERB.matcher(q);
            if (m.matches()) topic = m.group(1);
        }
        if (topic != null) {
            topic = TRAILING_VERB.matcher(topic).replaceFirst("").replaceFirst("^(the|a|an) ", "").trim();
            if (!topic.isEmpty()) out.add(topic);
            Matcher of = OF_TAIL.matcher(topic);  // "capital of spain" -> also "spain"
            if (of.find()) out.add(of.group(1));
        }
        if (out.isEmpty()) {  // anything else: its rarest-looking words
            List<String> words = new ArrayList<>();
            for (String w : q.split(" ")) if (Assistant.keywords(w).size() == 1 && words.size() < 3) words.add(w);
            if (!words.isEmpty()) out.add(Assistant.join(" ", words));
        }
        return new ArrayList<>(out);
    }
}
