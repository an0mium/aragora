'use client';

import type { UserPreferences } from './types';
import { ToggleSwitch } from './ToggleSwitch';

export interface NotificationsTabProps {
  preferences: UserPreferences;
  updateNotification: (key: keyof UserPreferences['notifications'], value: boolean) => void;
}

export function NotificationsTab({ preferences, updateNotification }: NotificationsTabProps) {
  return (
    <div className="card p-6" role="tabpanel" id="panel-notifications" aria-labelledby="tab-notifications">
      <h3 className="font-mono text-acid-green mb-4">Email Notifications</h3>
      <div className="space-y-4">
        <ToggleSwitch
          label="Debate Completed"
          description="Notify when a debate finishes"
          checked={preferences.notifications.debate_completed}
          onChange={() => updateNotification('debate_completed', !preferences.notifications.debate_completed)}
        />
        <ToggleSwitch
          label="Daily Digest"
          description="Summary of your debate activity"
          checked={preferences.notifications.email_digest}
          onChange={() => updateNotification('email_digest', !preferences.notifications.email_digest)}
        />
        <ToggleSwitch
          label="Weekly Summary"
          description="Weekly insights and trends"
          checked={preferences.notifications.weekly_summary}
          onChange={() => updateNotification('weekly_summary', !preferences.notifications.weekly_summary)}
        />
      </div>
    </div>
  );
}

export default NotificationsTab;
