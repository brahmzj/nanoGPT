package ai.morpheus;

import android.app.job.JobInfo;
import android.app.job.JobScheduler;
import android.content.ComponentName;
import android.content.Context;
import android.os.PersistableBundle;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 * The one Morpheus shared by the chat screen and the background learning job: which brain is
 * current, and a lock so only one learning session runs at a time.
 */
public final class Mind {

    private Mind() {}

    static final Object LEARNING = new Object();
    static final int JOB_ID = 4242, WORK_ID = 4243, NOW_ID = 4244;
    /** Busy right now (in this process): studying, or working on your tasks. */
    static volatile boolean studying, working, newWork;

    /** The chat screen, while you are looking at it: then news goes there instead of a notification. */
    interface Ui {
        void progress(String message);
        void studied(String summary, boolean kept);
        void news();
    }

    static volatile Ui ui;
    static Brain base;             // the brain that shipped inside the app
    static Brain current;          // what it thinks with now
    static long currentStamp = -1; // when learned.bin was last written, for the brain in `current`

    static File learnedFile(Context c) { return new File(c.getFilesDir(), "learned.bin"); }

    static synchronized Brain base(Context c) throws Exception {
        if (base == null) {
            try (InputStream in = c.getAssets().open("brain.bin")) {
                base = new Brain(in);
            }
        }
        return base;
    }

    /** The newest brain: the shipped one, or the one it grew into. */
    public static synchronized Brain brain(Context c) throws Exception {
        File f = learnedFile(c);
        long stamp = f.exists() ? f.lastModified() : 0;
        if (current == null || stamp != currentStamp) {
            current = stamp == 0 ? base(c) : trainer(c).toBrain();
            currentStamp = stamp;
        }
        return current;
    }

    public static synchronized boolean grewSince(Context c, Brain brain) throws Exception {
        return brain(c) != brain;
    }

    /** Rank of the adapters Morpheus learns in: its own brain stays frozen, so no skill is lost. */
    static final int RANK = 4;

    static Trainer trainer(Context c) throws Exception {
        File f = learnedFile(c);
        if (f.exists()) {
            try (InputStream in = new FileInputStream(f)) {
                return Trainer.load(base(c), in);
            } catch (Exception broken) {  // never let a bad file stop Morpheus from thinking
            }
        }
        Trainer fresh = new Trainer(base(c));
        fresh.useAdapters(RANK, System.currentTimeMillis());
        return fresh;
    }

    /** One learning session. Saves the new brain only if the session kept it. */
    public static Learner.Result learn(Context c, Phone phone, Learner.Progress progress, long budgetMillis) throws Exception {
        synchronized (LEARNING) {
            Trainer tr = trainer(c);
            Learner learner;
            try (InputStream lessons = c.getAssets().open("lessons.txt"); InputStream exam = c.getAssets().open("exam.txt")) {
                learner = new Learner(lessons, exam);
            }
            learner.lr = tr.rank > 0 ? 3e-3f : 1e-3f;
            Learner.Result r = learner.session(tr, phone, progress, 200, budgetMillis);
            if (r.kept) {
                File tmp = new File(c.getFilesDir(), "learned.tmp");
                try (FileOutputStream out = new FileOutputStream(tmp)) {
                    tr.save(out);
                }
                if (!tmp.renameTo(learnedFile(c))) throw new IllegalStateException("could not save what i learned");
            }
            return r;
        }
    }

    /** Learn every ~6 hours, only while charging (the energy comes from the wall, not the battery). */
    public static void schedule(Context c) {
        JobScheduler js = (JobScheduler) c.getSystemService(Context.JOB_SCHEDULER_SERVICE);
        if (js == null) return;
        for (JobInfo j : js.getAllPendingJobs()) if (j.getId() == JOB_ID) return;
        js.schedule(new JobInfo.Builder(JOB_ID, new ComponentName(c, LearnJob.class))
                        .setRequiresCharging(true)
                        .setPeriodic(6 * 3600 * 1000L)
                        .setPersisted(true)
                        .build());
    }

    /** Study now (🧠), in the background: it keeps going if you close the app, and tells you when it is done. */
    public static boolean studyNow(Context c) {
        JobScheduler js = (JobScheduler) c.getSystemService(Context.JOB_SCHEDULER_SERVICE);
        if (js == null || studying) return false;
        PersistableBundle now = new PersistableBundle();
        now.putInt("now", 1);
        try {
            studying = js.schedule(new JobInfo.Builder(NOW_ID, new ComponentName(c, LearnJob.class))
                                       .setExtras(now).setOverrideDeadline(0).build()) == JobScheduler.RESULT_SUCCESS;
        } catch (RuntimeException e) {
            studying = false;
        }
        return studying;
    }

    /** Work on your tasks in the background until they are done (see {@link WorkJob}). */
    public static void work(Context c) {
        if (working) { newWork = true; return; }  // the running job picks it up (rescheduling would stop it)
        JobScheduler js = (JobScheduler) c.getSystemService(Context.JOB_SCHEDULER_SERVICE);
        if (js == null) return;
        try {
            js.schedule(new JobInfo.Builder(WORK_ID, new ComponentName(c, WorkJob.class))
                            .setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY)
                            .setBackoffCriteria(20 * 60 * 1000L, JobInfo.BACKOFF_POLICY_LINEAR)
                            .setPersisted(true)
                            .build());
        } catch (RuntimeException e) {  // the tasks stay on its list: the next learning session works on them
        }
    }

    /** Tell you what it did: in the chat if you are looking at it, else a notification (only if there is news). */
    static void tell(Context c, Learner.Result r, String studied, boolean kept) {
        Ui u = ui;
        boolean news = !r.found.isEmpty() || !r.reports.isEmpty() || !r.stuck.isEmpty();
        if (u != null) {
            if (studied != null) u.studied(studied, kept);
            else if (news) u.news();
            return;
        }
        StringBuilder text = new StringBuilder();
        for (String f : r.found) text.append("i found out: ").append(f).append('\n');
        for (String line : r.reports) text.append(line).append('\n');
        for (String q : r.stuck) text.append("i could not find out: ").append(q).append(" do you know?\n");
        if (studied != null) text.append(studied);
        String title = !r.found.isEmpty() ? "Morpheus found out" + (r.found.size() > 1 ? " " + r.found.size() + " things" : "")
            : !r.stuck.isEmpty() ? "Morpheus needs your help" : studied != null ? "Morpheus finished studying"
            : !r.reports.isEmpty() ? "Morpheus finished reading" : null;
        if (title == null || text.length() == 0) return;
        Notify.show(c, studied != null ? NOW_ID : WORK_ID, title, text.toString().trim());
    }

    /** Forget everything learned on the phone and go back to the brain that shipped with the app. */
    public static synchronized void unlearn(Context c) {
        synchronized (LEARNING) {
            learnedFile(c).delete();
            current = null;
            currentStamp = -1;
        }
    }
}
