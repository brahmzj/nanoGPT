package ai.morpheus;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Connecting what it knows: a tiny knowledge graph from every sentence it has read or been told.
 *
 * "the eiffel tower is a famous iron tower in paris." becomes eiffel tower -is a-> tower and
 * eiffel tower -in-> paris. Chaining those answers questions no single sentence answers ("is the
 * eiffel tower in france?"), always with the reason. The world is open: when it cannot prove
 * something it says it does not know yet, never "no" (except that a place is in one country, one
 * city, one continent). The graph's edges also show what it does not know: France has a capital
 * and Spain is a country too, so what is the capital of Spain? Those gaps are its own questions.
 */
public final class Reasoner {

    static final class Fact {
        final String subject, relation, object, sentence;
        Fact(String subject, String relation, String object, String sentence) {
            this.subject = subject; this.relation = relation; this.object = object; this.sentence = sentence;
        }
    }

    public static final class Answer {
        public final String text;
        public final boolean proven;  // false: what it knows, and what it does not know yet
        Answer(String text, boolean proven) { this.text = text; this.proven = proven; }
    }

    final Map<String, List<Fact>> about = new LinkedHashMap<>();  // subject -> what it knows about it
    final Map<String, String> names = new HashMap<>();              // thing -> how it is written ("the eiffel tower")
    final Set<String> objects = new LinkedHashSet<>();              // things it has heard of

    public Reasoner(Iterable<String> sentences) {
        for (String s : sentences) read(s);
    }

    public int size() {
        int n = 0;
        for (List<Fact> fs : about.values()) n += fs.size();
        return n;
    }

    // ------------------------------------------------------------------ reading sentences into facts

    static final Pattern OF = Pattern.compile("^((?:the )?[a-z][a-z' -]{1,40}?) (?:is|are|was) the ([a-z ]{2,30}?) of ((?:the )?[a-z][a-z' -]{1,40})$");
    static final Pattern IN = Pattern.compile("^((?:the |a |an )?[a-z][a-z' -]{1,40}?) (?:is|are) in ((?:the )?[a-z][a-z' -]{1,40})$");
    static final Pattern ISA = Pattern.compile("^((?:the |a |an )?[a-z][a-z' -]{1,40}?) (?:is|are) (a |an )?([a-z][a-z' -]*)$");
    // where a noun phrase ends: "a famous iron tower | in paris", "a mountain | where lava ..."
    static final Pattern CUT = Pattern.compile(" (in|on|of|that|which|who|where|with|from|for|near|and|by|to|at|made|used|found|when|because|but|or|than)( |$)");
    static final Set<String> PLACES = new HashSet<>(Arrays.asList(
        "capital", "city", "town", "village", "port", "region", "state", "province", "island", "county", "district"));
    /** A place is directly in exactly one of each of these. */
    static final Set<String> EXCLUSIVE = new HashSet<>(Arrays.asList("country", "city", "continent", "state", "island", "town"));
    static final Set<String> VAGUE = new HashSet<>(Arrays.asList(
        "it", "they", "he", "she", "this", "that", "these", "those", "there", "which", "who", "one", "some", "many", "part",
        "kind", "type", "lot", "number", "name", "word", "way", "thing", "example", "place", "member", "form", "group",
        "north", "south", "east", "west", "middle", "center", "centre", "top", "bottom", "side", "edge", "area", "end"));
    static final Pattern IN_TAIL = Pattern.compile(" in ((?:the )?[a-z][a-z' -]{1,40})");

    void read(String sentence) {
        String s = sentence.toLowerCase(Locale.ROOT).trim().replaceAll("[.!?]+$", "").replaceAll("\\s+", " ");
        if (s.length() > 160) return;
        Matcher m;
        if ((m = OF.matcher(s)).matches()) {
            String x = m.group(1), y = m.group(3);
            for (String role : m.group(2).split(" and ")) {  // "the capital and largest city of france"
                role = role.replaceFirst("^the ", "").trim();
                String kind = last(role);
                add(x, "of:" + role, y, sentence);
                add(x, "isa", kind, sentence);
                if (PLACES.contains(kind)) add(x, "in", y, sentence);
            }
            return;
        }
        if ((m = IN.matcher(s)).matches()) {
            add(m.group(1), "in", cut(m.group(2)), sentence);
            return;
        }
        if ((m = ISA.matcher(s)).matches()) {
            String rest = m.group(3);
            if (m.group(2) == null && Reader.NOT_A_DEFINITION.matcher(rest).lookingAt()) return;  // "is on the seine"
            Matcher c = CUT.matcher(rest);
            String head = c.find() ? rest.substring(0, c.start()) : rest;
            // "lava is hot melted rock": a definition without "a" needs more than one word ("the sky is blue" is not one)
            if (m.group(2) == null && head.split(" ").length < 2) return;
            if (head.split(" ").length > 5) return;
            add(m.group(1), "isa", last(head), sentence);
            Matcher where = IN_TAIL.matcher(rest);
            if (where.find() && where.start() == head.length()) add(m.group(1), "in", cut(where.group(1)), sentence);
        }
    }

    static String cut(String phrase) {
        Matcher c = CUT.matcher(phrase);
        return (c.find() ? phrase.substring(0, c.start()) : phrase).replaceAll(",.*$", "").trim();
    }

    static String last(String phrase) {
        String[] w = phrase.trim().split(" ");
        return w[w.length - 1];
    }

    void add(String subject, String relation, String object, String sentence) {
        String x = thing(subject), y = thing(object);
        if (x.isEmpty() || y.isEmpty() || x.equals(y) || VAGUE.contains(x) || VAGUE.contains(y)) return;
        if (!names.containsKey(x)) names.put(x, subject.trim());
        if (!names.containsKey(y)) names.put(y, object.trim());
        List<Fact> fs = about.get(x);
        if (fs == null) about.put(x, fs = new ArrayList<>());
        for (Fact f : fs) if (f.relation.equals(relation) && f.object.equals(y)) return;
        fs.add(new Fact(x, relation, y, sentence));
        objects.add(y);
    }

    /** "the volcanoes" -> "volcano": the same thing however it is written. */
    static String thing(String phrase) {
        String p = phrase.toLowerCase(Locale.ROOT).trim().replaceFirst("^(the|a|an) ", "").trim();
        int sp = p.lastIndexOf(' ');
        String head = p.substring(sp + 1), rest = p.substring(0, sp + 1);
        return rest + singular(head);
    }

    static String singular(String w) {
        if (w.length() <= 3 || w.endsWith("ss") || w.endsWith("us") || w.endsWith("is")) return w;
        if (w.endsWith("ies")) return w.substring(0, w.length() - 3) + "y";
        if (w.endsWith("oes") || w.endsWith("ches") || w.endsWith("shes") || w.endsWith("xes")) return w.substring(0, w.length() - 2);
        if (w.endsWith("s")) return w.substring(0, w.length() - 1);
        return w;
    }

    String name(String thing) { return names.containsKey(thing) ? names.get(thing) : thing; }

    /** "a mountain", "the eiffel tower", "paris": how to say a thing in a sentence. */
    String say(String thing, boolean kind) {
        if (!kind) return name(thing);
        return ("aeiou".indexOf(thing.charAt(0)) >= 0 ? "an " : "a ") + thing;
    }

    List<Fact> facts(String x, String relation) {
        List<Fact> out = new ArrayList<>();
        List<Fact> fs = about.get(x);
        if (fs != null) for (Fact f : fs) if (f.relation.equals(relation)) out.add(f);
        return out;
    }

    /** Where x is: x, then what it is in, then what that is in... (at most 5 steps, no loops). */
    List<String> chain(String x) {
        List<String> path = new ArrayList<>(Arrays.asList(x));
        while (path.size() < 5) {
            List<Fact> up = facts(path.get(path.size() - 1), "in");
            String next = null;
            for (Fact f : up) if (!path.contains(f.object)) { next = f.object; break; }
            if (next == null) break;
            path.add(next);
        }
        return path;
    }

    /** A path of "is a" steps from x to kind, or null (breadth first, at most 3 steps). */
    List<String> kinds(String x, String kind) {
        List<List<String>> frontier = new ArrayList<>();
        frontier.add(new ArrayList<>(Arrays.asList(x)));
        Set<String> seen = new HashSet<>(Arrays.asList(x));
        for (int depth = 0; depth < 3; depth++) {
            List<List<String>> next = new ArrayList<>();
            for (List<String> path : frontier)
                for (Fact f : facts(path.get(path.size() - 1), "isa")) {
                    if (!seen.add(f.object)) continue;
                    List<String> p = new ArrayList<>(path);
                    p.add(f.object);
                    if (f.object.equals(kind)) return p;
                    next.add(p);
                }
            frontier = next;
        }
        return null;
    }

    static boolean covers(String place, String target) { return place.equals(target) || place.endsWith(" " + target); }

    // ------------------------------------------------------------------ answering

    static final Pattern IS_IN = Pattern.compile("^(?:is|are) (.+?) (?:in|a part of|part of) (.+?)\\??$");
    static final Pattern IS_A = Pattern.compile("^(?:is|are) (.+?) (?:a|an) (.+?)\\??$");
    static final Pattern ARE = Pattern.compile("^are (.+) ([a-z]+?)\\??$");  // "are volcanoes mountains?"
    static final Pattern WHERE = Pattern.compile("^where (?:is|are) (.+?)\\??$");

    /** An answer with its reasons, what it knows and does not know yet, or null if it knows nothing about it. */
    public Answer answer(String question) {
        String q = Assistant.key(question);
        Matcher m;
        if ((m = WHERE.matcher(q)).matches()) {
            List<String> path = chain(thing(m.group(1)));
            if (path.size() < 2) return null;
            StringBuilder sb = new StringBuilder(name(path.get(0)) + " is in " + name(path.get(1)));
            for (int i = 2; i < path.size(); i++) sb.append(", which is in ").append(name(path.get(i)));
            return new Answer(sb + ".", true);
        }
        if ((m = IS_IN.matcher(q)).matches()) {
            String x = thing(m.group(1)), target = thing(m.group(2));
            List<String> path = chain(x);
            if (path.size() < 2) return null;
            for (int i = 1; i < path.size(); i++)
                if (covers(path.get(i), target)) return new Answer("yes. " + steps(path.subList(0, i + 1), " is in ") + ".", true);
            for (int i = 1; i < path.size(); i++)  // a place is in only one country (city, continent...)
                for (Fact k : facts(path.get(i), "isa"))
                    if (EXCLUSIVE.contains(k.object) && isa(target, k.object) && !covers(target, path.get(i)))
                        return new Answer("no. " + steps(path.subList(0, i + 1), " is in ") + ", and " + name(path.get(i))
                                          + " and " + name(target) + " are different " + plural(k.object) + ".", true);
            return new Answer("i know " + steps(path.subList(0, 2), " is in ") + ", but not if " + pronoun(m.group(1))
                              + " in " + m.group(2) + ".", false);
        }
        if ((m = IS_A.matcher(q)).matches() || (m = ARE.matcher(q)).matches()) {
            String x = thing(m.group(1)), kind = thing(m.group(2));
            List<String> path = kinds(x, kind);
            if (path != null) {
                List<String> parts = new ArrayList<>();
                for (int i = 0; i + 1 < path.size(); i++)
                    parts.add((i == 0 ? name(path.get(0)) : say(path.get(i), true)) + " is " + say(path.get(i + 1), true));
                return new Answer("yes. " + joinAnd(parts) + ".", true);
            }
            List<Fact> known = facts(x, "isa");
            if (known.isEmpty()) return null;
            return new Answer("i know " + name(x) + " is " + say(known.get(0).object, true) + ", but not if "
                              + pronoun(m.group(1)) + " " + (q.startsWith("are ") && !q.contains(" a " + m.group(2)) && !q.contains(" an " + m.group(2))
                                  ? "" : q.contains(" an " + m.group(2)) ? "an " : "a ") + m.group(2) + ".", false);
        }
        return null;
    }

    boolean isa(String x, String kind) {
        for (Fact f : facts(x, "isa")) if (f.object.equals(kind)) return true;
        return false;
    }

    String steps(List<String> path, String link) {
        List<String> parts = new ArrayList<>();
        for (int i = 0; i + 1 < path.size(); i++) parts.add(name(path.get(i)) + link + name(path.get(i + 1)));
        return joinAnd(parts);
    }

    static String joinAnd(List<String> parts) {
        if (parts.size() == 1) return parts.get(0);
        return Assistant.join(", ", parts.subList(0, parts.size() - 1)) + " and " + parts.get(parts.size() - 1);
    }

    static String pronoun(String phrase) {
        return thing(phrase).equals(phrase.replaceFirst("^(the|a|an) ", "")) || !phrase.endsWith("s") ? "it is" : "they are";
    }

    static String plural(String kind) {
        if (kind.endsWith("y") && !kind.endsWith("ey")) return kind.substring(0, kind.length() - 1) + "ies";
        if (kind.endsWith("s") || kind.endsWith("sh") || kind.endsWith("ch")) return kind + "es";
        return kind + "s";
    }

    // ------------------------------------------------------------------ gaps: what it wants to know next

    /**
     * Questions its knowledge raises, most interesting first. Analogies: France is a country and
     * has a capital; Spain is a country too, so what is the capital of Spain? Then the edge of
     * what it knows: things it has heard of but knows nothing about ("what is a landform?").
     */
    public List<String> gaps(int n, Set<String> skip) {
        Set<String> out = new LinkedHashSet<>();
        Map<String, Set<String>> members = new HashMap<>();  // kind -> things of that kind
        for (List<Fact> fs : about.values())
            for (Fact f : fs) if (f.relation.equals("isa")) {
                Set<String> s = members.get(f.object);
                if (s == null) members.put(f.object, s = new LinkedHashSet<>());
                s.add(f.subject);
            }
        for (List<Fact> fs : about.values())  // 1. analogies
            for (Fact f : fs) {
                if (!f.relation.startsWith("of:")) continue;
                String role = f.relation.substring(3);
                for (Fact k : facts(f.object, "isa")) {
                    Set<String> same = members.get(k.object);
                    if (same == null) continue;
                    for (String other : same) if (!other.equals(f.object) && !hasRole(role, other))
                        out.add("what is the " + role + " of " + name(other));
                }
            }
        for (String y : objects)  // 2. heard of, but knows nothing about
            if (!about.containsKey(y) && !VAGUE.contains(y) && y.length() > 2)
                out.add("what " + (members.containsKey(y) ? "is " + say(y, true) : "is " + name(y)));
        List<String> list = new ArrayList<>();
        for (String q : out) if (list.size() < n && !skip.contains(q)) list.add(q);
        return list;
    }

    boolean hasRole(String role, String y) {
        for (List<Fact> fs : about.values()) for (Fact f : fs) if (f.relation.equals("of:" + role) && f.object.equals(y)) return true;
        return false;
    }
}
