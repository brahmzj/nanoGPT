package ai.morpheus;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Morpheus the assistant, in plain Java: a port of morpheus/assistant.py.
 *
 * Skills (notes, list, reminders, alarms, timers, clock, battery, checked answers, look-ups,
 * search) answer what they can exactly; everything else goes to the brain. Anything that
 * touches the phone goes through {@link Platform}, so this class is tested on a plain JVM.
 */
public final class Assistant {

    public static final String UNKNOWN_ANSWER = "i have not learned that yet. will you teach me?";

    /** What the assistant needs from the device. */
    public interface Platform {
        long now();                                        // epoch milliseconds
        List<String> readList(String name);
        void writeList(String name, List<String> items);
        boolean setAlarm(int hour, int minute, String message);
        boolean setTimer(int seconds, String message);
        boolean openUrl(String url);
        int[] battery();                                   // {percent, charging ? 1 : 0}, or null
        boolean internetAllowed();
        String[] lookup(String topic);                     // {title, extract}, or null
    }

    static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList((
        "a an the is are was were be my your our me i you it its of to in on at for and or what whats "
        + "where when who how which do does did that this these those please tell about can could would "
        + "remember forget there here have has had").split(" ")));

    final Brain brain;
    final Platform platform;
    final Conversation chat;
    final List<Object[]> skills = new ArrayList<>();  // {Pattern, skill name}

    public Assistant(Brain brain, Platform platform) {
        this.brain = brain;
        this.platform = platform;
        this.chat = new Conversation(brain);
        add("^(help|what can you do\\??|skills)$", "help");
        add("\\b(what time is it|what'?s the time|what is the time|tell me the time)\\b", "time");
        add("\\b(what day is (it|today)|what'?s (the |today'?s )?date|what is (the |today'?s )?date|what is today)\\b",
            "date");
        add("\\bbattery\\b", "battery");
        add("^remember (that )?(.+)$", "remember");
        add("^forget (that |about )?(.+)$", "forget");
        add("^what do you remember\\??$", "notes");
        add("^(add|put) (.+?) (to|on) (my |the )?(to-?do |todo |shopping )?list$", "todo add");
        add("^(remove|delete|cross off|check off) (.+?) (from|off) (my |the )?(to-?do |todo |shopping )?list$",
            "todo remove");
        add("^(done|finished|i did) (.+)$", "todo remove");
        add("^(what'?s|what is|show|read) (on )?(me )?(my |the )?(to-?do |todo |shopping )?list\\??$", "todo show");
        add("^clear (my |the )?(to-?do |todo |shopping )?list$", "todo clear");
        add("^set (a |an )?timer for (.+)$", "timer");
        add("^(set |make )?(an |a )?alarm (for|at) (.+)$", "alarm");
        add("^remind me (.+)$", "remind");
        add("^(search( the web)? for|google|search) (.+)$", "search");
        add("^(look up|lookup|tell me about|search wikipedia for|who is|who was) (.+?)\\??$", "lookup");
    }

    void add(String regex, String skill) { skills.add(new Object[]{Pattern.compile(regex), skill}); }

    /** Runs one skill; the regex groups each skill reads are listed with its pattern above. */
    String run(String skill, String text, Matcher m) {
        switch (skill) {
            case "help": return help();
            case "time": return time();
            case "date": return date();
            case "battery": return battery();
            case "remember": return remember(m.group(2));
            case "forget": return forget(m.group(2));
            case "notes": return listNotes();
            case "todo add": return todoAdd(m.group(2));
            case "todo remove": return todoRemove(text, m.group(2));
            case "todo show": return todoShow();
            case "todo clear": return todoClear();
            case "timer": return timer(m.group(2));
            case "alarm": return alarm(m.group(4));
            case "remind": return remind(m.group(1));
            case "search": return search(m.group(3));
            case "lookup": return lookup(text, m.group(2));
            default: throw new IllegalArgumentException(skill);
        }
    }

    // ------------------------------------------------------------------ entry point

    public String respond(String message) {
        String text = Alphabet.normalize(message).toLowerCase(Locale.ROOT).trim();
        text = text.replaceFirst("^(hey |hi |ok |okay )?morpheus[,!]?\\s+", "");
        if (text.isEmpty()) return "";
        List<String> notices = dueReminders();
        String reply = null;
        for (Object[] s : skills) {
            Matcher m = ((Pattern) s[0]).matcher(text);
            if (m.find()) {
                reply = run((String) s[1], text, m);
                if (reply != null) { chat.record(text, reply); break; }
            }
        }
        if (reply == null) {
            reply = taught(text);
            if (reply != null) chat.record(text, reply);
        }
        if (reply == null) {
            Object[] checked = check(text);
            if (checked != null) reply = checked(text, (String) checked[0], (Boolean) checked[1], (Boolean) checked[2]);
        }
        if (reply == null) {
            reply = chat.ask(text);
            if (reply.equals(UNKNOWN_ANSWER)) {
                String better = fallback(text);
                if (better != null) reply = better;
            }
        }
        notices.add(reply);
        return join("\n", notices);
    }

    String fallback(String text) {
        String note = recall(text);
        if (note != null) { chat.record(text, note); return note; }
        Matcher m = Pattern.compile("^(what|who) (is|are|was|were) (a |an |the )?(.+?)\\??$").matcher(text);
        if (m.matches() && platform.internetAllowed()) return lookup(text, m.group(4));
        return null;
    }

    // ------------------------------------------------------------------ taught facts

    /** Questions you taught with /teach are answered at once (the Termux version also trains them in). */
    public void teach(String question, String answer) {
        String q = key(question), a = Alphabet.normalize(answer).toLowerCase(Locale.ROOT).trim();
        List<String> facts = platform.readList("taught");
        for (java.util.Iterator<String> it = facts.iterator(); it.hasNext(); )
            if (it.next().startsWith(q + "\t")) it.remove();
        facts.add(q + "\t" + a);
        platform.writeList("taught", facts);
    }

    String taught(String text) {
        String q = key(text);
        for (String f : platform.readList("taught")) {
            int tab = f.indexOf('\t');
            if (tab > 0 && f.substring(0, tab).equals(q)) return f.substring(tab + 1);
        }
        return null;
    }

    static String key(String q) {
        return Alphabet.normalize(q).toLowerCase(Locale.ROOT).trim().replaceAll("[?.! ]+$", "");
    }

    // ------------------------------------------------------------------ skills

    String help() {
        return "i can answer what i learned (letters, numbers, math, words, colors, animals, days), "
            + "remember notes, keep your list, set reminders, timers and alarms, tell the time, "
            + "do any math, and look things up if you allow the internet.";
    }

    String time() { return "it is " + clock(platform.now()) + "."; }

    String date() {
        Date d = new Date(platform.now());
        String day = new SimpleDateFormat("EEEE, MMMM d, yyyy", Locale.US).format(d);
        return ("today is " + day + ".").toLowerCase(Locale.ROOT);
    }

    String battery() {
        int[] b = platform.battery();
        if (b == null) return "i can not see the battery from here.";
        return "the battery is at " + b[0] + "%" + (b[1] == 1 ? " and charging." : ".");
    }

    // notes

    String remember(String fact) {
        fact = fact.trim().replaceAll("\\.+$", "");
        List<String> notes = platform.readList("notes");
        notes.remove(fact);
        notes.add(fact);
        platform.writeList("notes", notes);
        return "ok, i will remember that " + secondPerson(fact) + ".";
    }

    String recall(String question) {
        Set<String> asked = keywords(question);
        if (asked.isEmpty()) return null;
        String best = null;
        double bestScore = 0;
        for (String note : platform.readList("notes")) {
            Set<String> k = keywords(note);
            k.retainAll(asked);
            double score = (double) k.size() / asked.size();
            if (!k.isEmpty() && score > bestScore) { best = note; bestScore = score; }
        }
        if (best == null || bestScore < 0.5) return null;
        return secondPerson(best) + ".";
    }

    String forget(String fact) {
        Set<String> target = keywords(fact);
        List<String> notes = platform.readList("notes"), keep = new ArrayList<>();
        for (String n : notes) {
            Set<String> k = keywords(n);
            k.retainAll(target);
            if (target.isEmpty() || k.size() < Math.max(1, target.size() / 2 + 1)) keep.add(n);
        }
        platform.writeList("notes", keep);
        int gone = notes.size() - keep.size();
        return gone > 0 ? "ok, i forgot " + gone + " note" + (gone != 1 ? "s" : "") + "." : "i did not have a note about that.";
    }

    String listNotes() {
        List<String> notes = platform.readList("notes");
        if (notes.isEmpty()) return "i have no notes yet. say: remember that ...";
        StringBuilder sb = new StringBuilder();
        int from = Math.max(0, notes.size() - 10);
        for (int i = from; i < notes.size(); i++)
            sb.append(i == from ? "" : " ").append(i - from + 1).append(") ").append(secondPerson(notes.get(i))).append(".");
        return sb.toString();
    }

    // to-do list

    String todoAdd(String item) {
        item = item.trim();
        List<String> items = platform.readList("todo");
        if (!items.contains(item)) { items.add(item); platform.writeList("todo", items); }
        return "added " + item + ". you have " + items.size() + " thing" + (items.size() != 1 ? "s" : "") + " on your list.";
    }

    String todoRemove(String text, String item) {
        item = item.trim().replaceAll("\\.+$", "");
        List<String> items = platform.readList("todo");
        String match = items.contains(item) ? item : null;
        if (match == null)
            for (String i : items) {
                Set<String> k = keywords(i);
                k.retainAll(keywords(item));
                if (!k.isEmpty()) { match = i; break; }
            }
        if (match == null)
            return text.startsWith("done") || text.startsWith("finished") || text.startsWith("i did")
                ? null : item + " is not on your list.";
        items.remove(match);
        platform.writeList("todo", items);
        return "removed " + match + ". " + (items.isEmpty() ? "your list is empty!" : items.size() + " left.");
    }

    String todoShow() {
        List<String> items = platform.readList("todo");
        return items.isEmpty() ? "your list is empty." : "on your list: " + join(", ", items) + ".";
    }

    String todoClear() {
        platform.writeList("todo", new ArrayList<>());
        return "your list is empty now.";
    }

    // reminders

    String timer(String when) {
        Integer seconds = parseDuration(when);
        if (seconds == null || seconds == 0) return "for how long? say: set a timer for 10 minutes.";
        return schedule(platform.now() + seconds * 1000L, "timer", seconds, null);
    }

    String alarm(String when) {
        int[] clock = parseClock(when);
        if (clock == null) return "at what time? say: set an alarm for 7:30 am.";
        return schedule(nextTime(clock), "alarm", null, clock);
    }

    String remind(String rest) {
        // {pattern, group of what, group of when, 1 = a duration / 0 = a clock time}
        Object[][] patterns = {{"^(to )?(.+?) in (.+)$", 2, 3, 1}, {"^in (.+?) to (.+)$", 2, 1, 1},
                               {"^(to )?(.+?) at (.+)$", 2, 3, 0}, {"^at (.+?) to (.+)$", 2, 1, 0}};
        for (Object[] p : patterns) {
            Matcher r = Pattern.compile((String) p[0]).matcher(rest);
            if (!r.matches()) continue;
            String what = r.group((Integer) p[1]).trim(), when = r.group((Integer) p[2]);
            if ((Integer) p[3] == 1) {
                Integer seconds = parseDuration(when);
                if (seconds != null && seconds > 0) return schedule(platform.now() + seconds * 1000L, what, seconds, null);
            } else {
                int[] clock = parseClock(when);
                if (clock != null) return schedule(nextTime(clock), what, null, clock);
            }
        }
        return "when should i remind you? say: remind me to call mom at 5pm, or in 20 minutes.";
    }

    long nextTime(int[] clock) {
        java.util.Calendar c = java.util.Calendar.getInstance();
        c.setTimeInMillis(platform.now());
        long now = c.getTimeInMillis();
        c.set(java.util.Calendar.HOUR_OF_DAY, clock[0]);
        c.set(java.util.Calendar.MINUTE, clock[1]);
        c.set(java.util.Calendar.SECOND, 0);
        c.set(java.util.Calendar.MILLISECOND, 0);
        if (c.getTimeInMillis() <= now) c.add(java.util.Calendar.DAY_OF_MONTH, 1);
        return c.getTimeInMillis();
    }

    String schedule(long due, String what, Integer seconds, int[] clock) {
        List<String> reminders = platform.readList("reminders");
        reminders.add(due + "\t" + what);
        platform.writeList("reminders", reminders);
        String label = "morpheus: " + what;
        boolean onPhone = seconds != null ? platform.setTimer(seconds, label) : platform.setAlarm(clock[0], clock[1], label);
        String where = onPhone ? " on your clock app" : "";
        if (what.equals("timer")) return "timer set for " + fmtDuration(seconds) + where + ".";
        if (what.equals("alarm")) return "alarm set for " + clock(due) + where + ".";
        return "ok, i will remind you to " + secondPerson(what) + " at " + clock(due) + where + ".";
    }

    List<String> dueReminders() {
        List<String> reminders = platform.readList("reminders"), keep = new ArrayList<>(), out = new ArrayList<>();
        long now = platform.now();
        for (String r : reminders) {
            int tab = r.indexOf('\t');
            long due = Long.parseLong(r.substring(0, tab));
            String what = r.substring(tab + 1);
            if (due <= now) {
                if (!what.equals("timer") && !what.equals("alarm")) out.add("(reminder: " + secondPerson(what) + ")");
            } else keep.add(r);
        }
        if (keep.size() != reminders.size()) platform.writeList("reminders", keep);
        return out;
    }

    // checked answers: the brain answers first, a tool checks it

    String checked(String text, String correct, boolean numeric, boolean learnable) {
        String attempt = chat.ask(text);
        if (numeric ? sameNumber(attempt, correct) : attempt.equals(correct)) return attempt;
        if (learnable) teach(text, correct);  // a mistake it could learn becomes a lesson
        chat.record(text, correct);
        return correct;
    }

    /** {correct answer, compare by final number?, learnable?}, or null if no tool can check this. */
    Object[] check(String text) {
        Matcher m = Pattern.compile("^which is (bigger|larger|greater|smaller|less), (-?\\d+) or (-?\\d+)\\??$")
            .matcher(text);
        if (m.matches()) {
            BigInteger a = new BigInteger(m.group(2)), b = new BigInteger(m.group(3));
            boolean bigger = !m.group(1).equals("smaller") && !m.group(1).equals("less");
            BigInteger v = bigger ? a.max(b) : a.min(b);
            boolean learnable = a.abs().max(b.abs()).compareTo(BigInteger.valueOf(100)) <= 0;
            return new Object[]{"the " + (bigger ? "bigger" : "smaller") + " of " + a + " and " + b + " is " + v + ".",
                                false, learnable};
        }
        m = Pattern.compile("^([a-z]+) has (\\d+) ([a-z]+) and gets (\\d+) more\\. "
                            + "how many ([a-z]+) now\\??$").matcher(text);
        if (m.matches()) {
            BigInteger total = new BigInteger(m.group(2)).add(new BigInteger(m.group(4)));
            String item = total.equals(BigInteger.ONE) ? m.group(3) : m.group(5);
            return new Object[]{m.group(1) + " has " + total + " " + item + ".", false,
                                total.compareTo(BigInteger.valueOf(40)) <= 0};
        }
        String expr = parseMath(text);
        if (expr == null) return null;
        Calc.Num value;
        try {
            value = new Calc(expr).parse();
        } catch (ArithmeticException e) {
            return new Object[]{"you can not divide by zero.", false, false};
        } catch (RuntimeException e) {
            return null;
        }
        Matcher nums = Pattern.compile("\\d+(?:\\.\\d+)?").matcher(expr);
        int count = 0;
        double max = 0;
        while (nums.find()) { count++; max = Math.max(max, Double.parseDouble(nums.group())); }
        boolean learnable = count == 2 && max <= 100 && value.isIntegral();
        return new Object[]{pretty(expr) + " = " + value.format() + ".", true, learnable};
    }

    // the outside world

    String search(String query) {
        query = query.trim();
        String url;
        try {
            url = "https://duckduckgo.com/?q=" + URLEncoder.encode(query, "UTF-8");
        } catch (java.io.UnsupportedEncodingException e) {
            throw new IllegalStateException(e);
        }
        return platform.openUrl(url) ? "searching for " + query + "." : "here is a search for " + query + ": " + url;
    }

    String lookup(String text, String topic) {
        topic = topic.trim().replaceAll("\\?+$", "").replaceFirst("^(a|an|the) ", "");
        if (!platform.internetAllowed())
            return "i have not learned about " + topic + " yet. if you allow the internet (tap the globe), i can look it up.";
        String[] found = platform.lookup(topic);
        if (found == null) return "i could not find anything about " + topic + ".";
        String answer = firstSentences(found[1], 220).toLowerCase(Locale.ROOT);
        if (answer.length() <= 90) teach(text, answer);
        return answer;
    }

    // ------------------------------------------------------------------ small parsers

    static String join(String sep, List<String> parts) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < parts.size(); i++) sb.append(i == 0 ? "" : sep).append(parts.get(i));
        return sb.toString();
    }

    static Set<String> keywords(String text) {
        Set<String> out = new HashSet<>();
        Matcher m = Pattern.compile("[a-z0-9']+").matcher(text.toLowerCase(Locale.ROOT));
        while (m.find()) if (!STOPWORDS.contains(m.group()) && m.group().length() > 1) out.add(m.group());
        return out;
    }

    static final Map<String, String> SWAPS = new HashMap<>();
    static {
        String[][] s = {{"my", "your"}, {"mine", "yours"}, {"i", "you"}, {"me", "you"}, {"myself", "yourself"},
                        {"i'm", "you're"}, {"am", "are"}, {"your", "my"}, {"yours", "mine"}};
        for (String[] p : s) SWAPS.put(p[0], p[1]);
    }

    static String secondPerson(String text) {
        List<String> out = new ArrayList<>();
        for (String w : text.trim().split("\\s+")) if (!w.isEmpty()) out.add((SWAPS.containsKey(w) ? SWAPS.get(w) : w));
        return join(" ", out);
    }

    static int[] parseClock(String text) {
        text = text.trim().toLowerCase(Locale.ROOT).replace(".", "");
        if (text.equals("noon") || text.equals("midday")) return new int[]{12, 0};
        if (text.equals("midnight")) return new int[]{0, 0};
        Matcher m = Pattern.compile("(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?").matcher(text);
        if (!m.matches()) return null;
        int hour = Integer.parseInt(m.group(1)), minute = m.group(2) == null ? 0 : Integer.parseInt(m.group(2));
        String half = m.group(3);
        if ("pm".equals(half) && hour < 12) hour += 12;
        if ("am".equals(half) && hour == 12) hour = 0;
        if (hour > 23 || minute > 59) return null;
        return new int[]{hour, minute};
    }

    static Integer parseDuration(String text) {
        text = text.toLowerCase(Locale.ROOT).replace("an hour", "1 hour").replace("a minute", "1 minute");
        Matcher m = Pattern.compile("(\\d+(?:\\.\\d+)?)\\s*(second|sec|minute|min|hour|hr)s?\\b").matcher(text);
        double total = 0;
        boolean found = false;
        while (m.find()) {
            String u = m.group(2);
            int unit = u.startsWith("sec") ? 1 : (u.startsWith("min") ? 60 : 3600);
            total += Double.parseDouble(m.group(1)) * unit;
            found = true;
        }
        return found ? (int) total : null;
    }

    static final String[][] WORD_OPS = {
        {"\\bmultiplied by\\b", "*"}, {"\\btimes\\b", "*"}, {"\\bdivided by\\b", "/"}, {"\\bover\\b", "/"},
        {"\\bplus\\b", "+"}, {"\\bminus\\b", "-"}, {"\\bto the power of\\b", "**"}, {"\\bsquared\\b", "**2"},
        {"\\bcubed\\b", "**3"}, {"(?<=[\\d)])\\s*x\\s*(?=[\\d(])", "*"}, {"\\^", "**"}, {"\\bmod\\b", "%"}};

    static String parseMath(String message) {
        String text = message.toLowerCase(Locale.ROOT).trim().replaceAll("[?.! ]+$", "");
        text = text.replaceFirst("^(what is|what's|whats|calculate|compute|how much is|solve)\\s+", "");
        text = text.replaceAll("(\\d+(?:\\.\\d+)?)\\s*% of\\s*", "$1/100*");
        for (String[] op : WORD_OPS) text = text.replaceAll(op[0], Matcher.quoteReplacement(op[1]));
        text = text.replace(",", "").replace("=", "").trim();
        if (!text.matches("[\\d\\s.+\\-*/%()]+") || !Pattern.compile("\\d\\s*[-+*/%]").matcher(text).find()) return null;
        return text.replaceAll("\\s+", "");
    }

    static String pretty(String expr) {
        return expr.replaceAll("(\\*\\*|[-+*/%])", " $1 ").replace(" * ", " x ").replace("  ", " ").trim();
    }

    static boolean sameNumber(String attempt, String correct) {
        String a = lastNumber(attempt), c = lastNumber(correct);
        return a != null && a.equals(c);
    }

    static String lastNumber(String s) {
        Matcher m = Pattern.compile("-?\\d+(?:\\.\\d+)?").matcher(s);
        String last = null;
        while (m.find()) last = m.group();
        return last;
    }

    static String clock(long millis) {
        String s = new SimpleDateFormat("hh:mm a", Locale.US).format(new Date(millis));
        return s.replaceFirst("^0", "").toLowerCase(Locale.ROOT);
    }

    static String fmtDuration(int seconds) {
        List<String> parts = new ArrayList<>();
        String[] names = {"hour", "minute", "second"};
        int[] sizes = {3600, 60, 1};
        for (int i = 0; i < 3; i++) {
            int n = seconds / sizes[i];
            seconds %= sizes[i];
            if (n > 0) parts.add(n + " " + names[i] + (n != 1 ? "s" : ""));
        }
        return parts.isEmpty() ? "0 seconds" : join(" and ", parts);
    }

    static String firstSentences(String text, int limit) {
        String out = "";
        for (String s : text.trim().split("(?<=[.!?])\\s+")) {
            if (!out.isEmpty() && out.length() + s.length() > limit) break;
            out = (out + " " + s).trim();
        }
        return out.length() > limit ? out.substring(0, limit) : out;
    }

    // ------------------------------------------------------------------ the conversation with the brain

    /** Recent turns as context, oldest dropped first, exactly like assistant.py's Conversation. */
    static final class Conversation {
        final Brain brain;
        final int reserve = 64;
        final List<String> turns = new ArrayList<>();

        Conversation(Brain brain) { this.brain = brain; }

        String context(List<String> turns, String turn) {
            List<String> all = new ArrayList<>(turns);
            all.add(turn);
            return "\n" + Assistant.join("\n", all);
        }

        String ask(String message) {
            message = Alphabet.normalize(message).toLowerCase(Locale.ROOT).trim();
            String turn = "user: " + message + "\nmorpheus: ";
            int budget = brain.blockSize() - reserve;
            while (!turns.isEmpty() && context(turns, turn).length() > budget) turns.remove(0);
            String context = context(turns, turn);
            if (context.length() > budget) context = context.substring(context.length() - budget);
            String reply = brain.generate(context, brain.blockSize() - context.length());
            turns.add(turn + reply);
            return reply;
        }

        void record(String message, String reply) {
            String turn = "user: " + Alphabet.normalize(message).toLowerCase(Locale.ROOT).trim() + "\nmorpheus: ";
            if (!turns.isEmpty() && turns.get(turns.size() - 1).startsWith(turn)) turns.set(turns.size() - 1, turn + reply);
            else turns.add(turn + reply);
        }

        void clear() { turns.clear(); }
    }

    public void reset() { chat.clear(); }

    // ------------------------------------------------------------------ a calculator with Python's number rules

    /** Parses + - * / // % ** and parentheses. Whole numbers stay exact (like Python ints). */
    static final class Calc {
        static final class Num {
            final BigInteger i;  // exact whole number, or null
            final double d;
            Num(BigInteger i) { this.i = i; this.d = i.doubleValue(); }
            Num(double d) { this.i = null; this.d = d; }
            boolean isIntegral() { return i != null || (d == Math.rint(d) && !Double.isInfinite(d)); }

            /** Python's fmt_number: exact ints; floats as whole numbers when they are, else 6 significant digits. */
            String format() {
                if (i != null) return i.toString();
                if (d == Math.rint(d) && Math.abs(d) < 1e15) return String.valueOf((long) d);
                if (Double.isNaN(d)) return "nan";
                if (Double.isInfinite(d)) return d > 0 ? "inf" : "-inf";
                BigDecimal b = new BigDecimal(d).round(new MathContext(6));
                int exp = b.precision() - b.scale() - 1;
                if (exp < -4 || exp >= 6) {
                    String mant = b.movePointLeft(exp).stripTrailingZeros().toPlainString();
                    return mant + "e" + (exp < 0 ? "-" : "+") + (Math.abs(exp) < 10 ? "0" : "") + Math.abs(exp);
                }
                return b.stripTrailingZeros().toPlainString();
            }
        }

        final String s;
        int p = 0;

        Calc(String s) { this.s = s; }

        Num parse() {
            Num v = expr();
            if (p != s.length()) throw new IllegalArgumentException("not arithmetic");
            return v;
        }

        boolean eat(String tok) {
            if (s.startsWith(tok, p)) { p += tok.length(); return true; }
            return false;
        }

        Num expr() {
            Num v = term();
            while (true) {
                if (eat("+")) v = add(v, term(), 1);
                else if (eat("-")) v = add(v, term(), -1);
                else return v;
            }
        }

        Num term() {
            Num v = unary();
            while (true) {
                if (s.startsWith("**", p)) return v;
                if (eat("//")) v = floorDiv(v, unary());
                else if (eat("*")) v = mul(v, unary());
                else if (eat("/")) v = div(v, unary());
                else if (eat("%")) v = mod(v, unary());
                else return v;
            }
        }

        Num unary() {
            if (eat("-")) { Num v = unary(); return v.i != null ? new Num(v.i.negate()) : new Num(-v.d); }
            if (eat("+")) return unary();
            return power();
        }

        Num power() {
            Num base = atom();
            if (eat("**")) {
                Num e = unary();
                if (Math.abs(e.d) > 100) throw new IllegalArgumentException("that power is too big for me");
                if (base.i != null && e.i != null && e.i.signum() >= 0) return new Num(base.i.pow(e.i.intValue()));
                return new Num(Math.pow(base.d, e.d));
            }
            return base;
        }

        Num atom() {
            if (eat("(")) {
                Num v = expr();
                if (!eat(")")) throw new IllegalArgumentException("missing )");
                return v;
            }
            int start = p;
            while (p < s.length() && (Character.isDigit(s.charAt(p)) || s.charAt(p) == '.')) p++;
            String n = s.substring(start, p);
            if (n.isEmpty()) throw new IllegalArgumentException("not arithmetic");
            return n.contains(".") ? new Num(Double.parseDouble(n)) : new Num(new BigInteger(n));
        }

        static Num add(Num a, Num b, int sign) {
            if (a.i != null && b.i != null) return new Num(sign > 0 ? a.i.add(b.i) : a.i.subtract(b.i));
            return new Num(sign > 0 ? a.d + b.d : a.d - b.d);
        }

        static Num mul(Num a, Num b) {
            if (a.i != null && b.i != null) return new Num(a.i.multiply(b.i));
            return new Num(a.d * b.d);
        }

        static Num div(Num a, Num b) {
            if (b.d == 0) throw new ArithmeticException("division by zero");
            return new Num(a.d / b.d);  // true division, like Python's /
        }

        static Num floorDiv(Num a, Num b) {
            if (b.d == 0) throw new ArithmeticException("division by zero");
            if (a.i != null && b.i != null) {
                BigInteger[] qr = a.i.divideAndRemainder(b.i);
                return new Num(qr[1].signum() != 0 && (qr[1].signum() != b.i.signum()) ? qr[0].subtract(BigInteger.ONE) : qr[0]);
            }
            return new Num(Math.floor(a.d / b.d));
        }

        static Num mod(Num a, Num b) {
            if (b.d == 0) throw new ArithmeticException("division by zero");
            if (a.i != null && b.i != null) {
                BigInteger r = a.i.mod(b.i.abs());
                return new Num(b.i.signum() < 0 && r.signum() != 0 ? r.add(b.i) : r);  // sign of the divisor
            }
            return new Num(a.d - b.d * Math.floor(a.d / b.d));
        }
    }
}
