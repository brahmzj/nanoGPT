package ai.morpheus;

import java.text.Normalizer;

/** Java twin of morpheus/tokenizer.normalize: fold any text down to the 96-symbol alphabet. */
public final class Alphabet {

    private Alphabet() {}

    public static String normalize(String text) {
        String s = Normalizer.normalize(text, Normalizer.Form.NFKD);
        s = s.replace("\r\n", "\n").replace('\r', '\n').replace('\t', ' ');
        StringBuilder out = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= 128) continue;  // accents, emoji ... are dropped, like encode("ascii", "ignore")
            out.append(Brain.CHARS.indexOf(c) >= 0 ? c : '?');
        }
        return out.toString();
    }

    /** The header stores the alphabet as hex so spaces and the newline survive. */
    static String decode(String hex) {
        if (hex == null) return null;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i + 1 < hex.length(); i += 2) sb.append((char) Integer.parseInt(hex.substring(i, i + 2), 16));
        return sb.toString();
    }
}
