package ai.morpheus;

import android.app.job.JobParameters;
import android.app.job.JobService;

/** Android starts this every ~6 hours while the phone is charging: one quiet learning session. */
public class LearnJob extends JobService {

    volatile boolean stopped;

    @Override
    public boolean onStartJob(final JobParameters params) {
        stopped = false;
        new Thread(new Runnable() {
            public void run() {
                try {
                    Mind.learn(getApplicationContext(), new Phone(getApplicationContext()), new Learner.Progress() {
                        public void update(String message) { }
                        public boolean cancelled() { return stopped; }
                    }, 7 * 60 * 1000L);  // Android gives a job about 10 minutes
                } catch (Throwable ignored) {
                    // a failed session changes nothing: the old brain stays
                }
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
