package ai.morpheus;

import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Handler;
import android.os.Looper;
import android.provider.AlarmClock;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/** The phone as Morpheus' hands and memory: storage, alarms, battery, the internet (when allowed). */
public final class Phone implements Assistant.Platform, Learner.World {

    final Context context;
    final SharedPreferences prefs;
    static final String WIKI = "https://simple.wikipedia.org";

    public Phone(Context context) {
        this.context = context;
        this.prefs = context.getSharedPreferences("morpheus", Context.MODE_PRIVATE);
    }

    public long now() { return System.currentTimeMillis(); }

    public synchronized List<String> readList(String name) {
        List<String> out = new ArrayList<>();
        try {
            JSONArray a = new JSONArray(prefs.getString("list_" + name, "[]"));
            for (int i = 0; i < a.length(); i++) out.add(a.getString(i));
        } catch (Exception ignored) {
        }
        return out;
    }

    public synchronized void writeList(String name, List<String> items) {
        prefs.edit().putString("list_" + name, new JSONArray(items).toString()).commit();
    }

    public boolean internetAllowed() { return prefs.getBoolean("internet", false); }

    // ------------------------------------------------------------------ the Clock app, the browser, the battery

    public boolean setAlarm(int hour, int minute, String message) {
        return start(new Intent(AlarmClock.ACTION_SET_ALARM).putExtra(AlarmClock.EXTRA_HOUR, hour)
                     .putExtra(AlarmClock.EXTRA_MINUTES, minute).putExtra(AlarmClock.EXTRA_MESSAGE, message)
                     .putExtra(AlarmClock.EXTRA_SKIP_UI, true));
    }

    public boolean setTimer(int seconds, String message) {
        return start(new Intent(AlarmClock.ACTION_SET_TIMER)
                     .putExtra(AlarmClock.EXTRA_LENGTH, Math.max(1, Math.min(seconds, 86400)))
                     .putExtra(AlarmClock.EXTRA_MESSAGE, message).putExtra(AlarmClock.EXTRA_SKIP_UI, true));
    }

    public boolean openUrl(String url) { return start(new Intent(Intent.ACTION_VIEW, Uri.parse(url))); }

    public int[] battery() {
        BatteryManager bm = (BatteryManager) context.getSystemService(Context.BATTERY_SERVICE);
        if (bm == null) return null;
        int percent = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
        return percent <= 0 ? null : new int[]{percent, bm.isCharging() ? 1 : 0};
    }

    /** Start an activity from any thread; true if the phone had an app for it. */
    boolean start(final Intent intent) {
        final boolean[] ok = {false};
        final CountDownLatch done = new CountDownLatch(1);
        Runnable job = new Runnable() {
            public void run() {
                try {
                    context.startActivity(intent);
                    ok[0] = true;
                } catch (ActivityNotFoundException | SecurityException e) {
                    ok[0] = false;
                }
                done.countDown();
            }
        };
        if (Looper.myLooper() == Looper.getMainLooper()) job.run();
        else new Handler(Looper.getMainLooper()).post(job);
        try {
            done.await(5, TimeUnit.SECONDS);
        } catch (InterruptedException ignored) {
        }
        return ok[0];
    }

    // ------------------------------------------------------------------ the internet, only when allowed

    public String[] lookup(String topic) {
        if (!internetAllowed()) return null;
        try {
            JSONArray hits = new JSONArray(get(WIKI + "/w/api.php?action=opensearch&format=json&limit=1&search="
                                               + URLEncoder.encode(topic, "UTF-8")));
            if (hits.getJSONArray(1).length() == 0) return null;
            String title = hits.getJSONArray(1).getString(0);
            return summary(WIKI + "/api/rest_v1/page/summary/" + URLEncoder.encode(title.replace(' ', '_'), "UTF-8"));
        } catch (Exception e) {
            return null;
        }
    }

    public String randomArticle() {
        if (!internetAllowed()) return null;
        try {
            String[] s = summary(WIKI + "/api/rest_v1/page/random/summary");
            return s == null ? null : s[0] + "\n" + s[1];
        } catch (Exception e) {
            return null;
        }
    }

    /** A whole Simple English Wikipedia article: {title, plain text, linked titles one per line}. */
    public String[] article(String topic) {
        if (!internetAllowed()) return null;
        try {
            JSONArray hits = new JSONArray(get(WIKI + "/w/api.php?action=opensearch&format=json&limit=1&search="
                                               + URLEncoder.encode(topic, "UTF-8")));
            if (hits.getJSONArray(1).length() == 0) return null;
            String title = hits.getJSONArray(1).getString(0);
            JSONObject pages = new JSONObject(get(WIKI + "/w/api.php?action=query&format=json&prop=extracts%7Clinks"
                + "&explaintext=1&redirects=1&plnamespace=0&pllimit=40&titles=" + URLEncoder.encode(title, "UTF-8")))
                .getJSONObject("query").getJSONObject("pages");
            JSONObject page = pages.getJSONObject(pages.keys().next());
            String text = page.optString("extract", "").replaceAll("(?m)^=+[^=]*=+\\s*$", " ");  // drop section headings
            if (text.trim().isEmpty()) return null;
            if (text.length() > 30_000) text = text.substring(0, 30_000);
            StringBuilder links = new StringBuilder();
            JSONArray ls = page.optJSONArray("links");
            for (int i = 0; ls != null && i < ls.length(); i++) links.append(ls.getJSONObject(i).optString("title")).append('\n');
            return new String[]{page.optString("title", title), text, links.toString()};
        } catch (Exception e) {
            return null;
        }
    }

    String[] summary(String url) throws Exception {
        JSONObject page = new JSONObject(get(url));
        String extract = page.optString("extract", "");
        return extract.isEmpty() ? null : new String[]{page.optString("title", ""), extract};
    }

    public String fetchText(String url) {
        if (!internetAllowed()) return null;
        try {
            String html = get(url);
            String text = html.replaceAll("(?is)<(script|style|noscript|head|nav|footer)[^>]*>.*?</\\1>", " ")
                .replaceAll("(?i)<(br|p|div|li|h[1-6]|tr)[^>]*>", "\n").replaceAll("<[^>]+>", " ")
                .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"")
                .replace("&#39;", "'").replace("&nbsp;", " ").replaceAll("[ \\t]+", " ").replaceAll("\\n\\s*\\n+", "\n");
            return text.length() > 200_000 ? text.substring(0, 200_000) : text;
        } catch (Exception e) {
            return null;
        }
    }

    static String get(String url) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(10000);
        c.setReadTimeout(15000);
        c.setRequestProperty("User-Agent", "Morpheus-Android/1.1 (tiny personal assistant)");
        try (InputStream in = c.getInputStream()) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            for (int n, total = 0; (n = in.read(buf)) > 0 && total < 2_000_000; total += n) out.write(buf, 0, n);
            return out.toString("UTF-8");
        } finally {
            c.disconnect();
        }
    }

    // ------------------------------------------------------------------ the library: everything read, compressed

    File libraryDir() {
        File d = new File(context.getFilesDir(), "library");
        d.mkdirs();
        return d;
    }

    static List<String> libraryCache;
    static String libraryStamp;

    public List<String> library() {
        List<String> out = new ArrayList<>();
        File[] files = libraryDir().listFiles();
        if (files == null) return out;
        Arrays.sort(files);
        String stamp = files.length + ":" + (files.length > 0 ? files[files.length - 1].getName() : "");
        synchronized (Phone.class) {
            if (stamp.equals(libraryStamp)) return new ArrayList<>(libraryCache);
        }
        int chars = 0;
        for (int i = files.length - 1; i >= 0 && chars < 1_000_000; i--) {  // newest first
            try (Reader r = new InputStreamReader(new GZIPInputStream(new FileInputStream(files[i])), "UTF-8")) {
                StringBuilder sb = new StringBuilder();
                char[] buf = new char[8192];
                for (int n; (n = r.read(buf)) > 0; ) sb.append(buf, 0, n);
                out.add(sb.toString());
                chars += sb.length();
            } catch (Exception ignored) {
            }
        }
        synchronized (Phone.class) {
            libraryCache = new ArrayList<>(out);
            libraryStamp = stamp;
        }
        return out;
    }

    public void addToLibrary(String text) {
        String clean = Alphabet.normalize(text).toLowerCase().replaceAll("[ ]+", " ").replaceAll("\\n\\s*\\n+", "\n").trim();
        if (clean.length() < 40) return;
        String id = Integer.toHexString(clean.hashCode());
        List<String> seen = readList("library_seen");
        if (seen.contains(id)) return;
        File f = new File(libraryDir(), System.currentTimeMillis() + "-" + id + ".txt.gz");
        try (Writer w = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(f)), "UTF-8")) {
            w.write(clean);
        } catch (Exception e) {
            return;
        }
        seen.add(id);
        writeList("library_seen", seen);
    }

    /** Share -> Morpheus: a link is read at the next session (if the internet is allowed), text right away. */
    public String share(String text) {
        String t = text.trim();
        if (t.matches("(?s)https?://\\S+")) {
            List<String> links = readList("shared_links");
            if (!links.contains(t)) links.add(t);
            writeList("shared_links", links);
            return internetAllowed() ? "thanks! i will read that page in my next learning session."
                                     : "thanks! tap the globe to let me read web pages, and i will read it in my next learning session.";
        }
        addToLibrary(t);
        return "thanks! i will read this in my next learning session.";
    }
}
