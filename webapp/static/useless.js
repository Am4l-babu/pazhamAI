/* Interactive parts of the Unnecessary Measurements suite:
   the end-detection confirmation loop, and the rotation counter. */
(function () {
  "use strict";

  // ── "Are you sure?" — a confirmation that confirms nothing ────────────────
  var reply = document.getElementById("sure-reply");
  var replies = [
    "Noted. The determination is unchanged.",
    "Your certainty has been recorded, then discarded.",
    "Re-examined at your request. It is still that end.",
    "That is the fourth time. It is still that end.",
    "The banana has no opinion on this matter.",
  ];
  var asked = 0;
  Array.prototype.forEach.call(document.querySelectorAll("[data-sure]"), function (btn) {
    btn.addEventListener("click", function () {
      if (!reply) return;
      reply.textContent = replies[Math.min(asked, replies.length - 1)];
      asked += 1;
    });
  });

  // ── Rotation counter — degrees accumulated, purpose not included ──────────
  var plate = document.getElementById("rot-plate");
  if (!plate) return;
  var arm = document.getElementById("rot-arm");
  var totalOut = document.getElementById("rot-total");
  var revOut = document.getElementById("rot-revs");

  var facing = 0;   // where the banana currently points
  var travelled = 0; // every degree ever turned, in either direction
  var last = null;

  function angleAt(event) {
    var box = plate.getBoundingClientRect();
    var dy = event.clientY - (box.top + box.height / 2);
    var dx = event.clientX - (box.left + box.width / 2);
    return Math.atan2(dy, dx) * 180 / Math.PI;
  }

  plate.addEventListener("pointerdown", function (event) {
    last = angleAt(event);
    plate.setPointerCapture(event.pointerId);
  });

  plate.addEventListener("pointermove", function (event) {
    if (last === null) return;
    var now = angleAt(event);
    var step = now - last;
    // A drag across the -180/180 seam is a small turn, not a full lap.
    while (step > 180) step -= 360;
    while (step < -180) step += 360;
    last = now;

    facing += step;
    travelled += Math.abs(step);
    arm.style.transform = "rotate(" + facing.toFixed(2) + "deg)";
    totalOut.textContent = travelled.toFixed(2) + "\u00B0";
    revOut.textContent = (travelled / 360).toFixed(3) + " revolutions of no consequence";
  });

  function release(event) {
    last = null;
    if (plate.hasPointerCapture && plate.hasPointerCapture(event.pointerId)) {
      plate.releasePointerCapture(event.pointerId);
    }
  }
  plate.addEventListener("pointerup", release);
  plate.addEventListener("pointercancel", release);
})();
