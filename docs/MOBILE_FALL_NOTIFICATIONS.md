# Mobile fall notifications

FusionSense can send a minimal phone notification through ntfy when the live
pipeline enters a validated fall state. This is a separate consumer of
`dashboard/live_output.json`; it cannot change model predictions or the
dashboard.

## Safety contract

A notification is eligible only when all of these are true:

- the source is the actively running `live_esp32` pipeline;
- the event is fresh (15 seconds old or less);
- the model emitted `FALL_ALERT` for `stand_to_fall` at or above its configured
  threshold;
- the IMU and camera pose were both valid for that inference window.

One fall episode sends one notification. Two subsequent valid `MONITORING`
windows re-arm the consumer. `DEGRADED` windows never send and never re-arm.
A 30-second delivery cooldown provides a second duplicate-alert barrier. State
is persisted in `dashboard/notification_state.json` across process restarts.

This remains a research prototype. It is not a clinical monitor or guaranteed
emergency service; Internet, laptop power, the ntfy service, and the phone must
all be available.

## Phone setup

1. Install the ntfy app on the receiving Android or iOS phone.
2. Choose a long, unguessable topic, for example
   `fusionsense-fall-<at-least-24-random-characters>`.
3. Subscribe to that exact topic in the phone app on `https://ntfy.sh`.

Public ntfy topics are readable by anyone who guesses the name, so never use a
person's name, phone number, address, or other identifying information in the
topic. A reserved authenticated topic or self-hosted ntfy server can instead
be configured with `FUSIONSENSE_NTFY_TOKEN` and `FUSIONSENSE_NTFY_SERVER`.

## Verify phone delivery

In PowerShell, set the topic only for the current terminal and send a harmless
setup notification:

```powershell
$env:FUSIONSENSE_NTFY_TOPIC = "your-long-random-topic"
.\.runtime\python311\python.exe .\scripts\send_mobile_notifications.py --test-notification
```

The phone should display `FusionSense mobile fall notifications are connected.`

## Run live

Keep the topic environment variable set and run these in separate terminals:

```powershell
.\.runtime\python311\python.exe -u .\scripts\run_live_fall_pipeline.py
```

```powershell
.\.runtime\python311\python.exe -u .\scripts\send_mobile_notifications.py
```

The dashboard server remains optional for phone delivery. Before testing a real
fall alert, use a safe staged action with another person present, a soft mat,
and no intentional uncontrolled fall.

For a non-network verification, use `--dry-run`; eligible alerts are printed
instead of transmitted. Use `--once` to examine only the current feed snapshot.
