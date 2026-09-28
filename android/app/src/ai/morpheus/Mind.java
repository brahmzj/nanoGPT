package ai.morpheus;

import android.app.job.JobInfo;
import android.app.job.JobScheduler;
import android.content.ComponentName;
import android.content.Context;

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
    static final int JOB_ID = 4242;
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

    /** Forget everything learned on the phone and go back to the brain that shipped with the app. */
    public static synchronized void unlearn(Context c) {
        synchronized (LEARNING) {
            learnedFile(c).delete();
            current = null;
            currentStamp = -1;
        }
    }
}
