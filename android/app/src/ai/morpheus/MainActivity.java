package ai.morpheus;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.Looper;
import android.provider.AlarmClock;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.text.InputType;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/** Morpheus on Android: a chat screen, the brain from assets/brain.bin, and the phone as its hands. */
public class MainActivity extends Activity implements Assistant.Platform {

    static final int NIGHT = Color.rgb(15, 11, 30), CARD = Color.rgb(30, 24, 56), LAVENDER = Color.rgb(237, 233, 254);
    static final int VIOLET = Color.rgb(109, 40, 217), DIM = Color.rgb(148, 140, 180), FIELD = Color.rgb(38, 31, 70);
    static final int VOICE = 7;

    final ExecutorService worker = Executors.newSingleThreadExecutor();
    SharedPreferences prefs;
    Assistant assistant;
    LinearLayout messages;
    ScrollView scroll;
    EditText input;
    Button globe, speaker;
    TextToSpeech tts;
    boolean speakReplies, heardByVoice;

    // ------------------------------------------------------------------ the screen

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        prefs = getSharedPreferences("morpheus", MODE_PRIVATE);
        speakReplies = prefs.getBoolean("speak", false);
        getWindow().setStatusBarColor(NIGHT);
        getWindow().setNavigationBarColor(NIGHT);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(NIGHT);

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setPadding(dp(18), dp(14), dp(10), dp(4));
        TextView title = text("☾ Morpheus", 22, LAVENDER);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        header.addView(title, new LinearLayout.LayoutParams(0, -2, 1));
        globe = iconButton("🌐");
        speaker = iconButton("🔊");
        Button reset = iconButton("↺");
        header.addView(globe);
        header.addView(speaker);
        header.addView(reset);
        root.addView(header);

        TextView subtitle = text("a tiny mind that grew up from abc and 123 · all on your phone", 13, DIM);
        subtitle.setPadding(dp(18), 0, dp(18), dp(10));
        root.addView(subtitle);

        scroll = new ScrollView(this);
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(dp(10), dp(6), dp(10), dp(6));
        scroll.addView(messages);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(10), dp(8), dp(10), dp(12));
        input = new EditText(this);
        input.setHint("ask morpheus…");
        input.setHintTextColor(DIM);
        input.setTextColor(LAVENDER);
        input.setTextSize(16);
        input.setSingleLine(true);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_AUTO_CORRECT);
        input.setImeOptions(EditorInfo.IME_ACTION_SEND);
        input.setPadding(dp(16), dp(10), dp(16), dp(10));
        input.setBackground(round(FIELD, 24));
        bar.addView(input, new LinearLayout.LayoutParams(0, -2, 1));
        Button mic = iconButton("🎤");
        Button send = iconButton("➤");
        bar.addView(mic);
        bar.addView(send);
        root.addView(bar);
        setContentView(root);

        send.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { heardByVoice = false; send(); }
        });
        mic.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { listen(); }
        });
        reset.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if (assistant != null) assistant.reset();
                messages.removeAllViews();
                bubble("(conversation forgotten. your notes and list are kept.)", false);
            }
        });
        globe.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                boolean on = !internetAllowed();
                prefs.edit().putBoolean("internet", on).apply();
                showToggles();
                toast(on ? "internet look-ups allowed" : "internet off: everything stays on your phone");
            }
        });
        speaker.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                speakReplies = !speakReplies;
                prefs.edit().putBoolean("speak", speakReplies).apply();
                showToggles();
                toast(speakReplies ? "i will speak my answers" : "i will stay quiet");
            }
        });
        input.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            public boolean onEditorAction(TextView v, int action, KeyEvent e) {
                if (action == EditorInfo.IME_ACTION_SEND || (e != null && e.getKeyCode() == KeyEvent.KEYCODE_ENTER)) {
                    heardByVoice = false;
                    send();
                    return true;
                }
                return false;
            }
        });
        showToggles();

        bubble("hello! i am morpheus. i learned from abc and 123. ask me about letters, numbers, "
               + "words or math, or say help. 🎤 lets you talk to me.", false);
        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            public void onInit(int status) { }
        });
        worker.execute(new Runnable() {
            public void run() {
                try {
                    InputStream in = getAssets().open("brain.bin");
                    final Brain brain = new Brain(in);
                    in.close();
                    assistant = new Assistant(brain, MainActivity.this);
                } catch (final Exception e) {
                    runOnUiThread(new Runnable() {
                        public void run() { bubble("(my brain would not load: " + e + ")", false); }
                    });
                }
            }
        });
    }

    @Override
    protected void onDestroy() {
        if (tts != null) tts.shutdown();
        worker.shutdown();
        super.onDestroy();
    }

    void send() {
        final String message = input.getText().toString().trim();
        if (message.isEmpty()) return;
        input.setText("");
        bubble(message, true);
        worker.execute(new Runnable() {
            public void run() {
                final String reply;
                final boolean unknown;
                if (assistant == null) {
                    reply = "(still waking up, try again in a moment)";
                    unknown = false;
                } else if (message.startsWith("/teach")) {
                    reply = teach(message.substring(6));
                    unknown = false;
                } else {
                    reply = assistant.respond(message);
                    unknown = reply.equals(Assistant.UNKNOWN_ANSWER);
                }
                runOnUiThread(new Runnable() {
                    public void run() {
                        bubble(reply, false);
                        if (speakReplies || heardByVoice) say(reply);
                        if (unknown) {  // one tap to teach: the question is ready, type the answer
                            String q = Alphabet.normalize(message).toLowerCase().trim();
                            input.setText("/teach " + q + " = ");
                            input.setSelection(input.getText().length());
                            hint("type the answer after = and send, and i will remember it.");
                        }
                    }
                });
            }
        });
    }

    String teach(String text) {
        int eq = text.indexOf('=');
        if (eq < 0) return "(to teach me: /teach what is the capital of france? = paris is the capital of france.)";
        String q = text.substring(0, eq).trim(), a = text.substring(eq + 1).trim();
        if (q.isEmpty() || a.isEmpty()) return "(i need both a question and an answer.)";
        assistant.teach(q, a);
        return "thank you! now i know: " + a;
    }

    void listen() {
        Intent i = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        i.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        i.putExtra(RecognizerIntent.EXTRA_PROMPT, "talk to morpheus");
        try {
            startActivityForResult(i, VOICE);
        } catch (ActivityNotFoundException e) {
            toast("this phone has no speech recognizer");
        }
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == VOICE && result == RESULT_OK && data != null) {
            ArrayList<String> heard = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
            if (heard != null && !heard.isEmpty()) {
                input.setText(heard.get(0));
                heardByVoice = true;  // a spoken question gets a spoken answer
                send();
            }
        }
    }

    void say(String text) {
        if (tts != null) tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "morpheus");
    }

    // ------------------------------------------------------------------ looks

    int dp(int n) { return (int) (n * getResources().getDisplayMetrics().density + 0.5f); }

    GradientDrawable round(int color, int radius) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(color);
        g.setCornerRadius(dp(radius));
        return g;
    }

    TextView text(String s, int sp, int color) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(sp);
        t.setTextColor(color);
        return t;
    }

    Button iconButton(String label) {
        Button b = new Button(this);
        b.setText(label);
        b.setTextSize(18);
        b.setTextColor(LAVENDER);
        b.setAllCaps(false);
        b.setBackground(round(Color.TRANSPARENT, 20));
        b.setMinWidth(dp(44));
        b.setMinimumWidth(dp(44));
        b.setPadding(dp(8), 0, dp(8), 0);
        return b;
    }

    void showToggles() {
        globe.setAlpha(internetAllowed() ? 1f : 0.35f);
        speaker.setText(speakReplies ? "🔊" : "🔈");
        speaker.setAlpha(speakReplies ? 1f : 0.5f);
    }

    void bubble(String s, boolean fromUser) {
        TextView t = text(s, 16, fromUser ? Color.WHITE : LAVENDER);
        t.setBackground(round(fromUser ? VIOLET : CARD, 18));
        t.setPadding(dp(14), dp(10), dp(14), dp(10));
        t.setMaxWidth((int) (getResources().getDisplayMetrics().widthPixels * 0.8));
        t.setTextIsSelectable(true);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-2, -2);
        lp.gravity = fromUser ? Gravity.END : Gravity.START;
        lp.setMargins(0, dp(4), 0, dp(4));
        messages.addView(t, lp);
        scrollDown();
    }

    void hint(String s) {
        TextView t = text(s, 13, DIM);
        t.setPadding(dp(6), 0, dp(6), dp(6));
        messages.addView(t);
        scrollDown();
    }

    void scrollDown() {
        scroll.post(new Runnable() {
            public void run() { scroll.fullScroll(View.FOCUS_DOWN); }
        });
    }

    void toast(String s) { Toast.makeText(this, s, Toast.LENGTH_SHORT).show(); }

    // ------------------------------------------------------------------ the phone as Morpheus' hands

    public long now() { return System.currentTimeMillis(); }

    public List<String> readList(String name) {
        List<String> out = new ArrayList<>();
        try {
            JSONArray a = new JSONArray(prefs.getString("list_" + name, "[]"));
            for (int i = 0; i < a.length(); i++) out.add(a.getString(i));
        } catch (Exception ignored) {
        }
        return out;
    }

    public void writeList(String name, List<String> items) {
        prefs.edit().putString("list_" + name, new JSONArray(items).toString()).apply();
    }

    public boolean setAlarm(int hour, int minute, String message) {
        final Intent i = new Intent(AlarmClock.ACTION_SET_ALARM)
            .putExtra(AlarmClock.EXTRA_HOUR, hour).putExtra(AlarmClock.EXTRA_MINUTES, minute)
            .putExtra(AlarmClock.EXTRA_MESSAGE, message).putExtra(AlarmClock.EXTRA_SKIP_UI, true);
        return start(i);
    }

    public boolean setTimer(int seconds, String message) {
        final Intent i = new Intent(AlarmClock.ACTION_SET_TIMER)
            .putExtra(AlarmClock.EXTRA_LENGTH, Math.max(1, Math.min(seconds, 86400)))
            .putExtra(AlarmClock.EXTRA_MESSAGE, message).putExtra(AlarmClock.EXTRA_SKIP_UI, true);
        return start(i);
    }

    public boolean openUrl(String url) {
        return start(new Intent(Intent.ACTION_VIEW, Uri.parse(url)));
    }

    public int[] battery() {
        BatteryManager bm = (BatteryManager) getSystemService(BATTERY_SERVICE);
        if (bm == null) return null;
        int percent = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
        if (percent <= 0) return null;
        return new int[]{percent, bm.isCharging() ? 1 : 0};
    }

    public boolean internetAllowed() { return prefs.getBoolean("internet", false); }

    public String[] lookup(String topic) {
        try {
            String base = "https://simple.wikipedia.org";
            JSONArray hits = new JSONArray(get(base + "/w/api.php?action=opensearch&format=json&limit=1&search="
                                               + URLEncoder.encode(topic, "UTF-8")));
            if (hits.getJSONArray(1).length() == 0) return null;
            String title = hits.getJSONArray(1).getString(0);
            JSONObject page = new JSONObject(get(base + "/api/rest_v1/page/summary/"
                                                 + URLEncoder.encode(title.replace(' ', '_'), "UTF-8")));
            String extract = page.optString("extract", "");
            return extract.isEmpty() ? null : new String[]{page.optString("title", title), extract};
        } catch (Exception e) {
            return null;
        }
    }

    static String get(String url) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(10000);
        c.setReadTimeout(10000);
        c.setRequestProperty("User-Agent", "Morpheus-Android/1.0 (tiny personal assistant)");
        try (InputStream in = c.getInputStream()) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            for (int n; (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
            return out.toString("UTF-8");
        } finally {
            c.disconnect();
        }
    }

    /** Start an activity from any thread; true if the phone had an app to handle it. */
    boolean start(final Intent intent) {
        return onUi(new Callable<Boolean>() {
            public Boolean call() {
                try {
                    startActivity(intent);
                    return true;
                } catch (ActivityNotFoundException | SecurityException e) {
                    return false;
                }
            }
        });
    }

    boolean onUi(final Callable<Boolean> job) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            try { return job.call(); } catch (Exception e) { return false; }
        }
        final boolean[] result = {false};
        final CountDownLatch done = new CountDownLatch(1);
        runOnUiThread(new Runnable() {
            public void run() {
                try { result[0] = job.call(); } catch (Exception ignored) { }
                done.countDown();
            }
        });
        try {
            done.await(5, TimeUnit.SECONDS);
        } catch (InterruptedException ignored) {
        }
        return result[0];
    }
}
