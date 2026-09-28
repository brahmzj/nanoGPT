package ai.morpheus;

import android.app.job.JobParameters;
import android.app.job.JobService;
import android.content.Context;

/**
 * Morpheus working on your tasks in the background (your questions, "find out ...", "learn about
 * ..."), with the app closed, until each one is done: found, read, or given up on after three ever
 * harder searches (then it asks you). Light work: a few pages per run, only with a network, only
 * while the internet is allowed. Android runs it again (20, 40, 60 minutes later) while tasks remain.
 */
public class WorkJob extends JobService {

    volatile boolean stopped;

    @Override
    public boolean onStartJob(final JobParameters params) {
        stopped = false;
        Mind.working = true;
        new Thread(new Runnable() {
            public void run() {
                Context c = getApplicationContext();
                Phone phone = new Phone(c);
                boolean more;
                try {
                    Learner.Result r = new Learner().research(phone, new Learner.Progress() {
                        public void update(String message) { }
                        public boolean cancelled() { return stopped; }
                    });
                    Mind.tell(c, r, null, false);
                    more = Learner.hasWork(phone);
                } catch (Throwable e) {
                    more = Learner.hasWork(phone);  // a bad moment (offline?): try again later
                }
                boolean fresh = Mind.newWork;  // a task you gave it while it was busy: start on it now, not in 20 minutes
                Mind.newWork = false;
                Mind.working = false;
                if (stopped) return;
                if (fresh && more) {
                    jobFinished(params, false);
                    Mind.work(c);
                } else {
                    jobFinished(params, more);
                }
            }
        }, "morpheus-working").start();
        return true;
    }

    @Override
    public boolean onStopJob(JobParameters params) {
        stopped = true;  // e.g. the network went away: Android runs it again later
        Mind.working = false;
        return true;
    }
}
