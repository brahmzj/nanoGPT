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
    /** How every "i don't know" starts: it is allowed not to know, never allowed to stop there. */
    public static final String DONT_KNOW = "i don't know yet";
    /**
     * Below this, the brain's answer is a guess. Measured on the 280 exam questions: right answers
     * have a median confidence of 1.00, wrong ones 0.59; under 0.7 catches 29 of the 32 wrong
     * answers and only 8 of the 248 right ones.
     */
    public static final float UNSURE = 0.7f;

    /** What the assistant needs from the device. */
    public interface Platform extends Curiosity.Store {
        long now();                                        // epoch milliseconds
        List<String> readList(String name);
        void writeList(String name, List<String> items);
        boolean setAlarm(int hour, int minute, String message);
        boolean setTimer(int seconds, String message);
        boolean openUrl(String url);
        int[] battery();                                   // {percent, charging ? 1 : 0}, or null
        boolean internetAllowed();
        String[] lookup(String topic);                     // {title, extract}, or null
        List<String> library();                            // everything it has read, newest first
        void addToLibrary(String text);
        void workInBackground();                           // keep working on your tasks until they are done
    }

    Reader.Memory memory;
    int memoryTexts = -1;

    /** A searchable memory of everything read, rebuilt when the reading grows. */
    Reader.Memory memory() {
        List<String> lib = platform.library();
        if (memory == null || lib.size() != memoryTexts) {
            memory = new Reader.Memory(lib);
            memoryTexts = lib.size();
        }
        return memory;
    }

    Reasoner reasoner;
    String reasonerStamp;

    /** Everything it read and was told, connected (rebuilt when either grows). */
    Reasoner reasoner() {
        Reader.Memory mem = memory();
        List<String> taught = platform.readList("taught");
        String stamp = memoryTexts + ":" + taught.size() + ":" + (taught.isEmpty() ? "" : taught.get(taught.size() - 1));
        if (reasoner == null || !stamp.equals(reasonerStamp)) {
            reasoner = new Reasoner(knownSentences());
            reasonerStamp = stamp;
        }
        return reasoner;
    }

    /** Every sentence it read, and every answer it was taught. */
    List<String> knownSentences() {
        List<String> sentences = new ArrayList<>(memory().sentences);
        for (String f : platform.readList("taught")) {
            int tab = f.indexOf('\t');
            if (tab > 0) sentences.add(f.substring(tab + 1));
        }
        return sentences;
    }

    static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList((
        "a an the is are was were be my your our me i you it its of to in on at for and or what whats "
        + "where when who how which do does did that this these those please tell about can could would "
        + "remember forget there here have has had").split(" ")));

    final Brain brain;
    final Platform platform;
    final Conversation chat;
    final Curiosity curiosity;
    final List<Object[]> skills = new ArrayList<>();  // {Pattern, skill name}
    /** The question it just asked you ("do you know?"): your next message may be the answer. */
    public volatile String asking;
    /** How sure it was of its last answer, and why. */
    String sure;

    public Assistant(Brain brain, Platform platform) {
        this.brain = brain;
        this.platform = platform;
        this.chat = new Conversation(brain);
        this.curiosity = new Curiosity(platform);
        add("^(help|what can you do\\??|skills)$", "help");
        add("\\b(what time is it|what'?s the time|what is the time|tell me the time)\\b", "time");
        add("\\b(what day is (it|today)|what'?s (the |today'?s )?date|what is (the |today'?s )?date|what is today)\\b",
            "date");
        add("\\bbattery\\b", "battery");
        add("^what (did|have) you (learn|learned|learnt)( lately| today| recently)?\\??$", "learned");
        add("^(what are you (curious|wondering) about|what do you wonder about|what do you want to know)\\??$", "wonders");
        add("^what (did|have) you (find|found|figure|figured) out( lately| today| recently)?\\??$", "found out");
        add("^(how sure are you|are you sure|how do you know( that)?)\\??$", "sure");
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
        add("^(learn|read|study|find out|research) (about|up on) (.+?)[.!?]*$", "interest");
        add("^(?:please |can you |could you |will you )?(?:find out|figure out|look into|research|investigate)"
            + " (?!about |up on )(.+?)[.!?]*$", "task");
        add("^(what are you (working on|doing)|what are your tasks|what tasks do you have)\\??$", "tasks");
        add("^(?:stop (?:working on|looking for|looking into|finding out|reading about)|never ?mind(?: about)?|cancel) (.+?)[.!?]*$",
            "drop");
        add("^(what are you (reading|learning|studying)( about)?|what are your interests)\\??$", "interests");
        add("^what do you know about (.+?)\\??$", "know");
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
            case "learned": return learned();
            case "wonders": return wonders();
            case "found out": return foundOut();
            case "sure": return sure == null ? "i have not answered anything yet." : sure;
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
            case "interest": return interest(m.group(3));
            case "task": return task(m.group(1));
            case "tasks": return tasks();
            case "drop": return drop(m.group(1));
            case "interests": return interests();
            case "know": return know(m.group(1));
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
        notices.addAll(curiosity.news());  // what it found out since you asked
        String pending = asking;
        asking = null;
        String reply = null;
        for (Object[] s : skills) {
            Matcher m = ((Pattern) s[0]).matcher(text);
            if (m.find()) {
                reply = run((String) s[1], text, m);
                if (reply != null) {
                    chat.record(text, reply);
                    if (!s[1].equals("sure")) sure = "sure: i did not have to guess.";
                    if (asking == null) asking = pending;  // a skill in between: its question still stands
                    break;
                }
            }
        }
        if (reply == null && pending != null) {  // it asked you something: is this the answer?
            reply = answered(pending, text);
            if (reply != null) chat.record(text, reply);
        }
        if (reply == null) {
            reply = taught(text);
            if (reply != null) { chat.record(text, reply); sure = "sure: i was taught that, or looked it up."; }
        }
        if (reply == null) {
            Object[] checked = check(text);
            if (checked != null) {
                reply = checked(text, (String) checked[0], (Boolean) checked[1], (Boolean) checked[2]);
                sure = "sure: i checked it (with a calculator, or letter by letter).";
            }
        }
        if (reply == null && isQuestion(text)) {  // connect what it knows: an answer it can prove, with the reason
            Reasoner.Answer r = reasoner().answer(text);
            if (r != null && r.proven) { reply = r.text; chat.record(text, reply); sure = "sure: i worked it out from what i know."; }
        }
        if (reply == null) reply = think(text);
        notices.add(reply);
        return join("\n", notices);
    }

    /**
     * The brain answers, and notices how sure it is (metacognition). Sure: it answers. Unsure: it
     * checks now if it may, or says it is not sure and wonders about it until it can check. It does
     * not know: it goes to find out.
     */
    String think(String text) {
        String reply = chat.ask(text);
        float confidence = brain.lastConfidence;
        if (reply.trim().isEmpty()) reply = UNKNOWN_ANSWER;  // saying nothing is not knowing
        // it was never taught a yes/no question: a reply without a yes or a no answered something else
        if (YES_NO.matcher(text).lookingAt() && !reply.matches("^(yes|no)\\b.*")) {
            Reasoner.Answer thought = reflect(text);
            if (thought != null && thought.proven) {
                chat.record(text, thought.text);
                sure = "sure: i thought it through from what i know.";
                return thought.text;
            }
            return dontKnow(text, thought);
        }
        if (reply.equals(UNKNOWN_ANSWER)) return dontKnow(text, null);
        String grounded = ground(text, reply);
        if (grounded != null) {  // what it read disagrees with a half-learned answer: trust the reading
            chat.record(text, grounded);
            sure = "sure: i read it.";
            return grounded;
        }
        int percent = Math.round(confidence * 100);
        sure = "about " + percent + "% sure.";
        if (confidence >= UNSURE || !isQuestion(text) || backed(text, reply)) return reply;
        if (platform.internetAllowed()) {
            String[] found = research(text);
            if (found != null) {
                chat.record(text, found[0]);
                sure = "sure now: i looked it up.";
                return "i was not sure, so i looked it up: " + found[0] + " (" + found[1] + ")";
            }
        }
        curiosity.wonder(text, reply, "unsure");
        sure = "not very: about " + percent + "% sure, so i will check.";
        return "i think " + reply + " but i am not sure yet, so i will check.";
    }

    static final Pattern IS_KIND = Pattern.compile("^(is|are) (.+?) (?:a|an) [a-z][a-z ]*$|^(are) ([a-z]+) [a-z]+$");
    static final Pattern IS_COLOR = Pattern.compile(
        "^(is|are) (.+?) (red|orange|yellow|green|blue|purple|pink|brown|black|white|gray|grey)$");

    /**
     * Think it through: a yes/no question it was never taught, turned into one it was ("is a cat an
     * animal?" -> "what is a cat?"). If it is sure of that answer, it reasons from it.
     */
    Reasoner.Answer reflect(String text) {
        String q = key(text);
        Matcher m = IS_COLOR.matcher(q);
        if (m.matches()) {
            String said = probe("what color " + m.group(1) + " " + m.group(2));
            if (said == null || !said.matches(".*\\b(red|orange|yellow|green|blue|purple|pink|brown|black|white|gray|grey)\\b.*")) return null;
            return new Reasoner.Answer(said.matches(".*\\b" + m.group(3) + "\\b.*") ? "yes. " + said : "i learned that " + said, true);
        }
        m = IS_KIND.matcher(q);
        if (!m.matches()) return null;
        String said = probe("what " + (m.group(1) != null ? m.group(1) + " " + m.group(2) : "are " + m.group(4)));
        if (said == null) return null;
        List<String> known = knownSentences();
        known.add(said);
        return new Reasoner(known).answer(q);
    }

    /** Ask its own brain a question it was taught, quietly (not part of the conversation); only a sure answer counts. */
    String probe(String question) {
        String said = brain.generate("\nuser: " + question + "?\nmorpheus: ", brain.blockSize() / 2);
        return said.isEmpty() || said.equals(UNKNOWN_ANSWER) || brain.lastConfidence < UNSURE ? null : said;
    }

    /** "I don't know yet" is allowed; stopping there is not. It looks, wonders, and asks. */
    String dontKnow(String text, Reasoner.Answer partial) {
        String better = fallback(text);  // notes, then reading
        if (better != null) { sure = "sure: i remembered it."; return better; }
        sure = "i did not know.";
        if (!isQuestion(text)) {
            String kept = told(text);
            return kept != null ? kept : UNKNOWN_ANSWER;
        }
        if (partial == null) partial = reasoner().answer(text);
        String known = partial != null ? partial.text + " " : "";
        curiosity.wonder(text, "", "you");
        if (platform.internetAllowed()) {
            String[] found = research(text);
            if (found != null) {
                chat.record(text, found[0]);
                sure = "sure now: i looked it up.";
                return known + "i did not know, so i looked it up: " + found[0] + " (" + found[1] + ")";
            }
            curiosity.tried(text);
            platform.workInBackground();
            asking = key(text);
            return known + DONT_KNOW + ". i looked, but could not find it yet. i will keep looking in the background, "
                + "and tell you when i find out. do you know?";
        }
        asking = key(text);
        return known + DONT_KNOW + ", but i want to find out. do you know? (or tap the globe, and i will look it up.)";
    }

    String fallback(String text) {
        String note = recall(text);
        if (note != null) { chat.record(text, note); return note; }
        String read = memory().recall(text);  // something it read
        if (read != null) { chat.record(text, read); return read; }
        return null;
    }

    /** Seek: look a question up right now (only if allowed). {answer, source}, or null. */
    String[] research(String question) {
        List<String> topics = Curiosity.topics(question);
        for (int i = 0; i < Math.min(2, topics.size()); i++) {
            String[] page = platform.lookup(topics.get(i));
            if (page == null) continue;
            String title = page[0].toLowerCase(Locale.ROOT);
            platform.addToLibrary(title + "\n" + page[1]);
            String hit = new Reader.Memory(Arrays.asList(title + "\n" + page[1])).recall(question);
            if (hit == null) continue;
            String answer = Reader.shorten(hit) != null ? Reader.shorten(hit) : hit, source = "simple wikipedia: " + title;
            if (answer.length() <= 120) learnFact(question, answer);
            curiosity.answered(question, answer, source);
            return new String[]{answer, source};
        }
        return null;
    }

    static final Pattern QUESTION = Pattern.compile(
        "^(what|what's|whats|who|whom|whose|where|when|why|how|which|is|are|was|were|am|do|does|did|can|could|will|would"
        + "|should|has|have|had)\\b.*|.*\\?$");

    static boolean isQuestion(String text) { return QUESTION.matcher(text.trim()).matches(); }

    static final Pattern YES_NO = Pattern.compile("^(is|are|was|were|am|do|does|did|can|could|will|would|should|has|have|had) ");

    static final Pattern NO_IDEA = Pattern.compile(
        "^(no|nope|nah|i don'?t know|i do not know|idk|no idea|not sure|i'?m not sure|i am not sure|never ?mind|skip|dunno)[.!]*$");
    static final Pattern NOT_AN_ANSWER = Pattern.compile(
        "^(ok|okay|k|thanks|thank you|cool|nice|great|hi|hello|hey|bye|yes|yeah|yep|lol|haha|hmm|huh|what|why|wow)[.!]*$");
    static final Pattern COMMAND = Pattern.compile(
        "^(tell|show|give|let|please|play|open|call|set|make|remind|add|help|say|sing|count|spell|teach|learn|read|stop|go)\\b");

    /** It asked you a question: a plain statement back is the answer, and it remembers it. */
    String answered(String question, String text) {
        if (NO_IDEA.matcher(text).matches()) return "that's ok! i will keep wondering, and look for it when i can.";
        if (isQuestion(text) || NOT_AN_ANSWER.matcher(text).matches() || COMMAND.matcher(text).lookingAt()) return null;
        String answer = text.replaceAll("\\s+", " ").trim();
        if (!answer.matches(".*[.!?]$")) answer += ".";
        teach(question, answer);
        sure = "sure: you told me.";
        return "thank you! now i know: " + answer;
    }

    /**
     * You told it something it did not know, like "madrid is the capital of spain.": if it is a fact
     * it can use, it keeps it, and notices when it answers one of its own questions.
     */
    String told(String text) {
        String answered = null;
        int kept = 0;
        List<Curiosity.Wonder> wonders = curiosity.all();
        for (String[] f : Reader.facts(null, text)) {
            if (f[0].matches(".*\\b(you|your|i|we|us|me|morpheus)\\b.*")) continue;  // about us, not the world
            learnFact(f[0], f[1]);
            kept++;
            for (Curiosity.Wonder w : wonders)
                if (w.question.equals(f[0]) && (w.status.equals("open") || w.status.equals("stuck"))) answered = f[0];
            curiosity.answered(f[0], f[1], "you told me");
        }
        if (kept == 0) return null;
        sure = "sure: you told me.";
        return answered != null ? "oh! that answers what i was wondering: " + answered + "? thank you!"
                                : "thank you! i will remember that.";
    }

    /** When you open the app: what it found out for you, and maybe a question of its own. */
    public String greet() {
        List<String> lines = curiosity.news();
        Curiosity.Wonder w = asking == null ? curiosity.toAsk(platform.internetAllowed()) : null;
        if (w != null) {
            asking = w.question;
            lines.add("i have been wondering: " + w.question + "? " + (w.tries > 0 ? "i could not find it. " : "") + "do you know?");
        }
        return join("\n", lines);
    }

    // ------------------------------------------------------------------ taught facts

    /** Questions you taught with /teach are answered at once, and studied in the next learning session. */
    public void teach(String question, String answer) {
        learnFact(question, answer);
        curiosity.answered(question, Alphabet.normalize(answer).toLowerCase(Locale.ROOT).trim(), "you told me");
    }

    void learnFact(String question, String answer) {
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

    // ------------------------------------------------------------------ learning

    /** Is its answer what its reading says? Then it can be sure, however unsure the brain felt. */
    boolean backed(String question, String reply) {
        List<String> hits = memory().search(question, 5);
        Set<String> said = keywords(reply);
        said.removeAll(keywords(question));
        if (hits.isEmpty() || said.isEmpty()) return false;
        Set<String> support = new HashSet<>();
        for (String h : hits) support.addAll(keywords(h));
        int backed = 0;
        for (String w : said) if (support.contains(w)) backed++;
        return backed * 2 >= said.size();
    }

    /**
     * Grounding: if its reading covers the question but does not support the brain's answer (a fact
     * only half learned comes out garbled), answer with the sentence it read instead. The brain's
     * attempt still helps pick which sentence: the one sharing the most words with it.
     */
    String ground(String question, String reply) {
        List<String> hits = memory().search(question, 5);
        if (hits.isEmpty()) return null;
        Set<String> said = keywords(reply);
        said.removeAll(keywords(question));
        if (said.isEmpty()) return null;
        Set<String> support = new HashSet<>();
        for (String h : hits) support.addAll(keywords(h));
        int backed = 0;
        for (String w : said) if (support.contains(w)) backed++;
        if (backed * 2 >= said.size()) return null;  // the brain agrees with what it read
        if (question.matches("^(what|who) (is|are|was|were) .*")) return hits.get(0);  // a definition wins
        String best = hits.get(0);
        int bestShared = -1;
        for (String h : hits) {
            Set<String> k = keywords(h);
            k.retainAll(said);
            if (k.size() > bestShared) { bestShared = k.size(); best = h; }
        }
        return best;
    }

    /** "learn about volcanoes": a task. It reads about it in the background, and studies it while you charge it. */
    String interest(String topic) {
        topic = topic.trim().replaceFirst("^(a|an|the) ", "");
        List<String> queue = platform.readList("reading_queue");
        for (java.util.Iterator<String> it = queue.iterator(); it.hasNext(); )
            if (it.next().startsWith(topic + "\t")) it.remove();
        queue.add(0, topic + "\t0");
        platform.writeList("reading_queue", queue);
        if (!platform.internetAllowed())
            return "ok! tap the globe so i can read on the internet, and i will start reading about " + topic + ".";
        platform.workInBackground();
        return "ok! i am reading about " + topic + " in the background. i will tell you when i am done, "
            + "and study it the next time you charge me.";
    }

    /** "find out who invented the telephone": a task it keeps working on, in the background, until it is done. */
    String task(String what) {
        String q = key(what).replaceFirst("^(for me |please )", "");
        if (!isQuestion(q)) q = "what is " + q.replaceFirst("^(a|an|the) ", "the ");  // "find out the capital of peru"
        String known = taught(q);
        if (known == null) known = memory().recall(q);
        if (known != null) return "i already know: " + known;
        curiosity.wonder(q, "", "you");
        if (!platform.internetAllowed()) {
            asking = q;
            return "ok! i will find out " + q + "? as soon as you tap the globe (i need the internet for that). or do you know?";
        }
        String[] found = research(q);
        if (found != null) return "i found out: " + found[0] + " (" + found[1] + ")";
        platform.workInBackground();
        return "ok! i am on it: " + q + "? i will keep working on it in the background until i find out, and tell you.";
    }

    String tasks() {
        List<String> doing = new ArrayList<>(), help = new ArrayList<>();
        for (Curiosity.Wonder w : curiosity.tasks()) {
            if (w.status.equals("stuck")) help.add(w.question + "?");
            else doing.add("finding out " + w.question + "?" + (w.tries > 0 ? " (i looked " + w.tries + " time" + (w.tries == 1 ? "" : "s") + ")" : ""));
        }
        for (String q : platform.readList("reading_queue")) if (q.endsWith("\t0")) doing.add("reading about " + q.split("\t")[0] + ".");
        if (doing.isEmpty() && help.isEmpty()) return "nothing right now. give me something to find out, or say: learn about volcanoes.";
        String out = doing.isEmpty() ? "" : "i am working on: " + join(" ", doing)
            + (platform.internetAllowed() ? "" : " (i need the internet for that: tap the globe.)");
        if (!help.isEmpty()) out += " i need your help with: " + join(" ", help) + " do you know?";
        return out.trim();
    }

    String drop(String what) {
        int n = curiosity.drop(what);
        List<String> queue = platform.readList("reading_queue"), keep = new ArrayList<>();
        for (String q : queue) if (!q.split("\t")[0].equals(what.replaceFirst("^(a|an|the) ", ""))) keep.add(q);
        n += queue.size() - keep.size();
        platform.writeList("reading_queue", keep);
        return n > 0 ? "ok, i stopped working on that." : "i was not working on that.";
    }

    String interests() {
        List<String> want = new ArrayList<>(), done = platform.readList("read_titles");
        for (String q : platform.readList("reading_queue")) { if (want.size() < 5) want.add(q.split("\t")[0]); }
        if (want.isEmpty() && done.isEmpty()) return "nothing yet. say: learn about volcanoes.";
        String out = want.isEmpty() ? "" : "i want to read about " + join(", ", want) + ". ";
        if (!done.isEmpty()) out += "lately i read about " + join(", ", done.subList(Math.max(0, done.size() - 5), done.size())) + ".";
        return out.trim();
    }

    String know(String topic) {
        List<String> found = memory().search("what is " + topic, 2);
        if (found.isEmpty()) return "i have not read about " + topic + " yet. say: learn about " + topic + ".";
        return join(" ", found);
    }


    String learned() {
        List<String> sessions = platform.readList("sessions");
        if (sessions.isEmpty())
            return "i have not had a learning session yet. i study while you charge me, or tap the brain to teach me now.";
        return sessions.get(sessions.size() - 1);
    }

    String wonders() {
        List<String> open = new ArrayList<>(), stuck = new ArrayList<>();
        for (Curiosity.Wonder w : curiosity.mostInteresting(5, Curiosity.known(platform)))
            open.add(w.question + "?" + (w.origin.equals("itself") ? " (my own question)" : ""));
        for (Curiosity.Wonder w : curiosity.all()) if (w.status.equals("stuck") && stuck.size() < 3) stuck.add(w.question + "?");
        if (open.isEmpty() && stuck.isEmpty()) return "nothing right now. ask me things i do not know!";
        String out = open.isEmpty() ? "" : "i wonder: " + join(" ", open)
            + (platform.internetAllowed() ? " i will look them up when i study." : " tap the globe, and i will look them up when i study.");
        if (!stuck.isEmpty()) out += " i could not find out: " + join(" ", stuck) + " do you know?";
        asking = key((open.isEmpty() ? stuck.get(0) : open.get(0)).replaceFirst("\\?.*$", ""));  // an answer is welcome
        return out.trim();
    }

    String foundOut() {
        List<String> out = new ArrayList<>();
        List<Curiosity.Wonder> all = curiosity.all();
        for (int i = all.size() - 1; i >= 0 && out.size() < 3; i--) {
            Curiosity.Wonder w = all.get(i);
            if ((w.status.equals("found") || w.status.equals("told")) && !w.answer.isEmpty() && !w.source.equals("you told me"))
                out.add(w.question + "? " + w.answer);
        }
        return out.isEmpty() ? "nothing yet. ask me things i do not know, and i will try to find out."
                             : "i found out: " + join(" ", out);
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
        if (learnable) learnFact(text, correct);  // a mistake it could learn becomes a lesson
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
        m = Pattern.compile("^how do you spell ([a-z]+)\\??$").matcher(text);
        if (m.matches()) {  // spelling is checked letter by letter, the way you check your own copying
            String w = m.group(1);
            return new Object[]{w + " is spelled " + join(" ", Arrays.asList(w.split(""))) + ".", false, w.length() <= 12};
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
        String read = memory().recall(text);  // what it already read comes first
        if (read != null) return read;
        if (!platform.internetAllowed()) {
            curiosity.wonder(text, "", "you");  // looked up as soon as it may
            return "i have not learned about " + topic + " yet. if you allow the internet (tap the globe), i can look it up.";
        }
        String[] found = platform.lookup(topic);
        if (found == null) {
            curiosity.wonder(text, "", "you");
            curiosity.tried(text);
            return "i could not find anything about " + topic + " yet. i will keep looking.";
        }
        platform.addToLibrary(found[0].toLowerCase(Locale.ROOT) + "\n" + found[1]);
        String answer = firstSentences(found[1], 220).toLowerCase(Locale.ROOT);
        if (answer.length() <= 90) learnFact(text, answer);
        return answer;
    }

    // ------------------------------------------------------------------ small parsers

    static String join(String sep, List<String> parts) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < parts.size(); i++) sb.append(i == 0 ? "" : sep).append(parts.get(i));
        return sb.toString();
    }

    public static Set<String> keywordsOf(String text) { return keywords(text); }

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
