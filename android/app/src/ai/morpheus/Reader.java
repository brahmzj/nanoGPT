package ai.morpheus;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * How Morpheus absorbs what it reads.
 *
 * A 0.7M-parameter brain that reads raw text learns words and style, not facts it can answer.
 * So reading is turned into two things: questions with answers (from simple sentences like
 * "paris is the capital of france." -> "what is the capital of france?") that it studies in its
 * learning sessions, and a searchable memory of every sentence, for recall when its brain does
 * not know the answer (retrieval, the way big assistants look things up).
 */
public final class Reader {

    private Reader() {}

    static final int MAX_ANSWER = 100;

    /** Clean text into lowercase sentences, resolving "it"/"they" at the start to the title. */
    public static List<String> sentences(String title, String text) {
        String s = Alphabet.normalize(text).toLowerCase(Locale.ROOT)
            .replaceAll("\\([^()]*\\)", " ")    // (asides)
            .replaceAll("\\[[^\\]]*\\]", " ")   // [1] references
            .replaceAll("\\s+", " ").trim();
        String subject = title == null ? null : Alphabet.normalize(title).toLowerCase(Locale.ROOT).replaceAll("\\s*\\(.*$", "").trim();
        List<String> out = new ArrayList<>();
        String last = subject == null || subject.isEmpty() ? null : subject;  // "it" at the very start means the title
        for (String sentence : s.split("(?<=[.!?])\\s+")) {
            sentence = sentence.trim().replaceAll("\\s+([,.;:!?])", "$1");
            // "it"/"they" mean the previous sentence's subject ("the eiffel tower ... in paris. it is 330
            // metres tall." is about the tower, not paris). If that is unclear, the pronoun stays.
            Matcher pro = Pattern.compile("^(it|they|he|she) (is|are|was|were|has|have|had|lives|live) ").matcher(sentence);
            if (pro.lookingAt() && last != null) sentence = last + " " + sentence.substring(pro.end(1) + 1);
            Matcher subj = SUBJECT.matcher(sentence);
            last = subj.lookingAt() && words(subj.group(1)) <= 5
                   && !subj.group(1).matches("(it|they|he|she|this|that|there|these|those)") ? subj.group(1) : null;
            if (sentence.length() >= 12 && sentence.length() <= 220) out.add(sentence);
        }
        return out;
    }

    static final Pattern SUBJECT = Pattern.compile(
        "^((?:the |a |an )?[a-z][a-z' -]{1,40}?) (is|are|was|were|has|have|had|lives|live) ");

    // predicates that say something about X but do not define it: passives, places, "best known for"
    static final Pattern NOT_A_DEFINITION = Pattern.compile(
        "(in|on|at|near|from|by|born|built|founded|made|discovered|invented|written|created|started|found|known|called"
        + "|named|used|located|best|also|often|usually|still|now|very|not|[a-z]+ed) ");
    static final Pattern PERSON = Pattern.compile("\\b(physicist|scientist|writer|author|poet|king|queen|president|singer"
        + "|actor|actress|painter|artist|person|man|woman|leader|inventor|philosopher|composer|musician|politician"
        + "|player|footballer|explorer|emperor|prince|princess|chemist|biologist|mathematician|teacher|doctor)\\b");

    // "X is/are/was/were ..." with a short subject
    static final Pattern DEFINE = Pattern.compile("^((?:the |a |an )?[a-z][a-z' -]{1,40}?) (is|are|was|were) (.+)$");
    static final Pattern OF = Pattern.compile("^(?:the )?(.{2,40}?) (is|are|was|were) the ([a-z ]{2,30}?) of (.{2,40})$");
    static final Pattern HAS = Pattern.compile("^((?:the |a |an )?[a-z][a-z' -]{1,40}?) (has|have|had) (\\d[\\d,]*) ([a-z]+)(.*)$");
    static final Pattern BORN = Pattern.compile("^([a-z][a-z' .-]{1,40}?) was born (in|on) (.+)$");
    static final Pattern MADE = Pattern.compile(
        "^((?:the |a |an )?[a-z][a-z' -]{1,40}?) (was|were) (built|founded|made|discovered|invented|written|created|started) (in|by) (.+)$");
    static final Pattern WHERE = Pattern.compile("^((?:the |a |an )?[a-z][a-z' -]{1,40}?) (is|are) (in|on|near) (.+)$");

    /** Questions and answers Morpheus can study, from one article. */
    public static List<String[]> facts(String title, String text) {
        Map<String, String> out = new LinkedHashMap<>();
        for (String sentence : sentences(title, text)) {
            String body = sentence.replaceAll("[.!?]+$", "");
            String answer = shorten(sentence);
            if (answer == null) continue;
            Matcher m;
            if ((m = OF.matcher(body)).matches() && words(m.group(1)) <= 5)
                for (String role : m.group(3).split(" and "))  // "the capital and largest city of france"
                    put(out, "what " + m.group(2) + " the " + role.replaceFirst("^the ", "").trim() + " of " + m.group(4), answer);
            if ((m = HAS.matcher(body)).matches() && words(m.group(1)) <= 5)
                put(out, "how many " + m.group(4) + " " + (m.group(2).equals("had") ? "did" : m.group(2).equals("has") ? "does" : "do")
                         + " " + m.group(1) + " have", answer);
            if ((m = BORN.matcher(body)).matches() && words(m.group(1)) <= 4) {
                boolean date = m.group(2).equals("on") || m.group(3).matches("(\\d|january|february|march|april|may|june|july"
                                                                            + "|august|september|october|november|december).*");
                put(out, (date ? "when" : "where") + " was " + m.group(1) + " born", answer);
            }
            if ((m = MADE.matcher(body)).matches() && words(m.group(1)) <= 5) {
                boolean by = m.group(4).equals("by");
                put(out, by ? "who " + (m.group(3).equals("written") ? "wrote" : m.group(3)) + " " + m.group(1)
                            : "when " + m.group(2) + " " + m.group(1) + " " + m.group(3), answer);
            }
            if ((m = WHERE.matcher(body)).matches() && words(m.group(1)) <= 5)
                put(out, "where " + m.group(2) + " " + m.group(1), answer);
            if ((m = DEFINE.matcher(body)).matches() && words(m.group(1)) <= 4 && !NOT_A_DEFINITION.matcher(m.group(3)).lookingAt()
                && !m.group(1).matches(".*\\b(this|that|these|those|there|which|who|it|they|he|she)\\b.*")) {
                put(out, "what " + m.group(2) + " " + m.group(1), answer);
                if (PERSON.matcher(m.group(3)).find()) put(out, "who " + m.group(2) + " " + m.group(1), answer);
                // "mount everest is the tallest mountain": "the tallest mountain" names one thing, so it works both ways
                if (m.group(3).startsWith("the ") && words(m.group(3)) <= 7 && !m.group(3).matches(".*\\b(of|who|which|that|where|when|and|or|but)\\b.*"))
                    put(out, (PERSON.matcher(m.group(3)).find() ? "who " : "what ") + m.group(2) + " " + m.group(3), answer);
            }
        }
        List<String[]> list = new ArrayList<>();
        for (Map.Entry<String, String> e : out.entrySet()) list.add(new String[]{e.getKey(), e.getValue()});
        return list;
    }

    static void put(Map<String, String> out, String question, String answer) {
        question = question.replaceAll("\\s+", " ").trim();
        if (question.length() <= 70 && !out.containsKey(question)) out.put(question, answer);
    }

    static int words(String s) { return s.trim().split("\\s+").length; }

    /** Keep answers short enough to learn: cut long sentences at a clause break. */
    static String shorten(String sentence) {
        if (sentence.length() <= MAX_ANSWER) return sentence.endsWith(".") ? sentence : sentence + ".";
        for (String cut : new String[]{", which ", ", and ", " which ", ", ", " and ", " that "}) {
            int i = sentence.indexOf(cut, 25);
            if (i > 0 && i <= MAX_ANSWER - 1) return sentence.substring(0, i) + ".";
        }
        return null;
    }

    // ------------------------------------------------------------------ recall: a searchable memory of everything read

    /** Keyword search over sentences, weighting rare words more (a tiny BM25-like score). */
    public static final class Memory {
        final List<String> sentences = new ArrayList<>();
        final List<Set<String>> keys = new ArrayList<>();
        final Map<String, Integer> df = new HashMap<>();

        public Memory(List<String> texts) {
            for (String text : texts) {
                String title = firstLine(text);  // library pages are "title \n text"
                String body = title == null ? text : text.substring(text.indexOf('\n') + 1);
                for (String s : sentences(title, body)) {
                    if (sentences.size() >= 30000) break;
                    Set<String> k = Assistant.keywords(s);
                    if (k.isEmpty()) continue;
                    sentences.add(s);
                    keys.add(k);
                    for (String w : k) df.put(w, df.containsKey(w) ? df.get(w) + 1 : 1);
                }
            }
        }

        static String firstLine(String text) {
            int nl = text.indexOf('\n');
            return nl > 0 && nl < 60 ? text.substring(0, nl) : null;
        }

        public int size() { return sentences.size(); }

        /** The best matching sentence, or null if nothing covers enough of the question. */
        public String recall(String question) {
            List<String> best = search(question, 1);
            return best.isEmpty() ? null : best.get(0);
        }

        /** Up to n sentences that cover enough of the question, best first. */
        public List<String> search(String question, int n) {
            List<String> out = new ArrayList<>();
            Set<String> q = Assistant.keywords(question);
            if (q.isEmpty() || sentences.isEmpty()) return out;
            double total = 0;
            Map<String, Double> weight = new HashMap<>();
            for (String w : q) {
                int d = df.containsKey(w) ? df.get(w) : 0;
                double idf = Math.log(1 + (sentences.size() + 1.0) / (d + 0.5));
                weight.put(w, idf);
                total += idf;
            }
            // "what is a volcano?" is best answered by a sentence that starts with the volcano
            Matcher def = Pattern.compile("^(what|who) (is|are|was|were) (?:a |an |the )?(.+?)\\??$")
                .matcher(question.toLowerCase(Locale.ROOT).trim());
            Pattern about = def.matches() ? Pattern.compile("^(?:a |an |the )?" + Pattern.quote(def.group(3)) + "s? ") : null;
            final double[] score = new double[sentences.size()];
            List<Integer> good = new ArrayList<>();
            for (int i = 0; i < sentences.size(); i++) {
                for (String w : q) if (keys.get(i).contains(w)) score[i] += weight.get(w);
                score[i] -= 0.002 * sentences.get(i).length();  // prefer the shorter of equal matches
                if (about != null && score[i] >= 0.6 * total && about.matcher(sentences.get(i)).lookingAt()) score[i] += total;
                if (score[i] >= 0.6 * total) good.add(i);
            }
            java.util.Collections.sort(good, new java.util.Comparator<Integer>() {
                public int compare(Integer a, Integer b) { return Double.compare(score[b], score[a]); }
            });
            for (int i : good) {
                if (out.size() >= n) break;
                if (!out.contains(sentences.get(i))) out.add(sentences.get(i));
            }
            return out;
        }
    }
}
