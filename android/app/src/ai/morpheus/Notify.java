package ai.morpheus;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;

/** A notification when Morpheus finishes a task while you are away. Tapping it opens the chat. */
final class Notify {

    private Notify() {}

    static final String CHANNEL = "morpheus";

    static void show(Context c, int id, String title, String text) {
        NotificationManager nm = (NotificationManager) c.getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm == null) return;
        Intent open = new Intent(c, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        Notification.Builder b = new Notification.Builder(c)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(new Notification.BigTextStyle().bigText(text))
            .setAutoCancel(true)
            .setContentIntent(PendingIntent.getActivity(c, id, open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE));
        if (Build.VERSION.SDK_INT >= 26) {  // Android 8+ needs a channel (built against API 23, so by name)
            try {
                Class<?> ch = Class.forName("android.app.NotificationChannel");
                Object channel = ch.getConstructor(String.class, CharSequence.class, int.class)
                    .newInstance(CHANNEL, "What Morpheus found out", 3 /* IMPORTANCE_DEFAULT */);
                NotificationManager.class.getMethod("createNotificationChannel", ch).invoke(nm, channel);
                Notification.Builder.class.getMethod("setChannelId", String.class).invoke(b, CHANNEL);
            } catch (Exception ignored) {
            }
        }
        try {
            nm.notify(id, b.build());
        } catch (SecurityException noPermission) {  // not allowed to notify: the news waits in the chat
        }
    }
}
