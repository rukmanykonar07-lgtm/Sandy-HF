// Minimal service worker -- its only job right now is to exist, since a
// registered service worker is what makes a page installable as a PWA.
// No offline caching: Sandy needs the live backend to do anything useful
// anyway, so caching pages for offline use would be pretend-functionality.
// Revisit this file if/when real offline behavior is ever wanted.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
self.addEventListener("fetch", () => {
  // Intentionally not intercepting requests yet.
});
