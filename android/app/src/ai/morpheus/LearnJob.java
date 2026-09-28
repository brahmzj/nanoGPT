package ai.morpheus;

import android.app.job.JobParameters;
import android.app.job.JobService;
import android.content.Context;

/**
 * A learning session in the background: every ~6 hours while the phone is charging, or right now
 * when you tap 🧠 (then it keeps going if you close the app, and tells you when it is done).
 */
public class LearnJob extends JobService {

    volatile boolean stopped;

    @Override
    public boolean onStartJob(final JobParameters params) {
        stopped = false;
        final boolean now = params.getExtras() != null && params.getExtras().getInt("now", 0) == 1;
        if (!now && Mind.studying) return false;  // already studying: once is enough
        Mind.studying = true;
        new Thread(new Runnable() {
            public void run() {
                Context c = getApplicationContext();
                Learner.Result r = null;
                String summary;
                try {
                    r = Mind.learn(c, new Phone(c), new Learner.Progress() {
                        public void update(String message) {
                            Mind.Ui u = Mind.ui;
                            if (u != null) u.progress(message);
                        }
                        public boolean cancelled() { return stopped; }
                    }, (now ? 8 : 7) * 60 * 1000L);  // Android gives a job about 10 minutes
                    summary = r.summary;
                } catch (Throwable e) {
                    summary = "(i could not study: " + e + ")";  // a failed session changes nothing: the old brain stays
                }
                Mind.studying = false;
                if (r == null) r = new Learner.Result();
                Mind.tell(c, r, now || Mind.ui != null ? summary : null, r.kept);
                jobFinished(params, false);
            }
        }, "morpheus-learning").start();
        return true;
    }

    @Override
    public boolean onStopJob(JobParameters params) {
        stopped = true;  // e.g. unplugged: stop, keep the old brain, try again next time
        return true;
    }
}
