package ai.morpheus;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
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
public class MainActivity extends Activity {

    static final int NIGHT = Color.rgb(15, 11, 30), CARD = Color.rgb(30, 24, 56), LAVENDER = Color.rgb(237, 233, 254);
    static final int VIOLET = Color.rgb(109, 40, 217), DIM = Color.rgb(148, 140, 180), FIELD = Color.rgb(38, 31, 70);
    static final int VIOLET_LIGHT = Color.rgb(196, 181, 253);
    static final int VOICE = 7;

    final ExecutorService worker = Executors.newSingleThreadExecutor();
    SharedPreferences prefs;
    Phone phone;
    volatile Assistant assistant;
    volatile Brain brainInUse;
    volatile boolean learning;
    TextView status;
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
        phone = new Phone(this);
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
        Button brainButton = iconButton("🧠");
        header.addView(brainButton);
        globe = iconButton("🌐");
        speaker = iconButton("🔊");
        Button reset = iconButton("↺");
        header.addView(globe);
        header.addView(speaker);
        header.addView(reset);
        root.addView(header);

        TextView subtitle = text("a tiny mind that grew up from abc and 123 · all on your phone", 13, DIM);
        subtitle.setPadding(dp(18), 0, dp(18), dp(4));
        root.addView(subtitle);
        status = text("", 12, VIOLET_LIGHT);
        status.setPadding(dp(18), 0, dp(18), dp(8));
        root.addView(status);

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
        brainButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { learnNow(); }
        });
        brainButton.setOnLongClickListener(new View.OnLongClickListener() {
            public boolean onLongClick(View v) { askToUnlearn(); return true; }
        });
        globe.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                boolean on = !phone.internetAllowed();
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
               + "words or math, or say help. 🎤 lets you talk to me. i keep learning while your phone "
               + "charges, and 🧠 makes me study right now. to let me read and teach myself, tap 🌐 "
               + "and say: learn about volcanoes.", false);
        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            public void onInit(int status) { }
        });
        handleShare(getIntent());
        Mind.schedule(this);
        worker.execute(new Runnable() {
            public void run() {
                try {
                    brainInUse = Mind.brain(MainActivity.this);
                    assistant = new Assistant(brainInUse, phone);
                } catch (final Exception e) {
                    runOnUiThread(new Runnable() {
                        public void run() { bubble("(my brain would not load: " + e + ")", false); }
                    });
                }
            }
        });
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleShare(intent);
    }

    @Override
    protected void onResume() {
        super.onResume();
        worker.execute(new Runnable() {  // did it study while we were away?
            public void run() {
                try {
                    if (brainInUse != null && !learning && Mind.grewSince(MainActivity.this, brainInUse)) {
                        brainInUse = Mind.brain(MainActivity.this);
                        assistant = new Assistant(brainInUse, phone);
                        List<String> sessions = phone.readList("sessions");
                        final String last = sessions.isEmpty() ? "" : sessions.get(sessions.size() - 1);
                        runOnUiThread(new Runnable() {
                            public void run() { bubble("(while you were away, i studied. " + last + ")", false); }
                        });
                    }
                } catch (Exception ignored) {
                }
            }
        });
    }

    void handleShare(Intent intent) {
        if (intent == null || !Intent.ACTION_SEND.equals(intent.getAction())) return;
        String text = intent.getStringExtra(Intent.EXTRA_TEXT);
        String subject = intent.getStringExtra(Intent.EXTRA_SUBJECT);
        if (text == null || text.trim().isEmpty()) return;
        intent.setAction(Intent.ACTION_MAIN);  // handle each share once
        bubble("(shared" + (subject != null ? ": " + subject : "") + ")", true);
        bubble(phone.share((subject != null && !text.startsWith("http") ? subject + "\n" : "") + text), false);
    }

    void learnNow() {
        if (learning) { toast("i am already studying"); return; }
        learning = true;
        status.setText("🧠 getting ready to study… (you can keep talking to me)");
        new Thread(new Runnable() {
            public void run() {
                String message;
                try {
                    Learner.Result r = Mind.learn(MainActivity.this, phone, new Learner.Progress() {
                        public void update(final String m) {
                            runOnUiThread(new Runnable() { public void run() { status.setText("🧠 " + m); } });
                        }
                        public boolean cancelled() { return isFinishing(); }
                    }, 8 * 60 * 1000L);
                    message = r.summary;
                    if (r.kept) {
                        brainInUse = Mind.brain(MainActivity.this);
                        assistant = new Assistant(brainInUse, phone);
                    }
                } catch (Throwable e) {
                    message = "(i could not study: " + e + ")";
                }
                final String done = message;
                learning = false;
                runOnUiThread(new Runnable() {
                    public void run() {
                        status.setText("");
                        bubble(done, false);
                        if (speakReplies) say(done);
                    }
                });
            }
        }, "morpheus-learn-now").start();
    }

    void askToUnlearn() {
        new AlertDialog.Builder(this)
            .setTitle("Go back to the original brain?")
            .setMessage("Morpheus forgets what it learned on this phone. Your notes, list and taught answers are kept.")
            .setPositiveButton("Go back", new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface d, int which) {
                    worker.execute(new Runnable() {
                        public void run() {
                            try {
                                Mind.unlearn(MainActivity.this);
                                brainInUse = Mind.brain(MainActivity.this);
                                assistant = new Assistant(brainInUse, phone);
                            } catch (Exception ignored) {
                            }
                        }
                    });
                    bubble("(back to the brain i shipped with.)", false);
                }
            })
            .setNegativeButton("Cancel", null)
            .show();
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
                } else if (message.equals("/learn")) {
                    runOnUiThread(new Runnable() { public void run() { learnNow(); } });
                    reply = "ok, i will study now.";
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
        globe.setAlpha(phone.internetAllowed() ? 1f : 0.35f);
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

}
