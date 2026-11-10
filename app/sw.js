/* Service worker: pre-caches the whole app (HTML, JS, TF.js library, model) for Airplane Mode.
   After changing any file or re-exporting the model, bump CACHE so phones download the new version. */
const CACHE = 'leaf-check-v3';
const CORE = [
  './',
  'index.html',
  'app.js',
  'manifest.json',
  'vendor/tf.min.js',
  'model/model.json',
  'model/metadata.json',
  'icons/icon-192.png',
  'icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    // cache: 'reload' skips the browser's HTTP cache so a new version never re-caches stale files
    const fresh = (urls) => urls.map((u) => new Request(u, { cache: 'reload' }));
    await cache.addAll(fresh(CORE));
    // The weight shard names (group1-shard1of2.bin, ...) are listed inside model.json
    const manifest = await (await cache.match('model/model.json')).json();
    const shards = manifest.weightsManifest.flatMap((g) => g.paths.map((p) => 'model/' + p));
    await cache.addAll(fresh(shards));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

// Cache-first: serve from cache, fall back to the network (and store what we fetch).
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith((async () => {
    const cached = await caches.match(event.request, { ignoreSearch: true });
    if (cached) return cached;
    try {
      const response = await fetch(event.request);
      if (response.ok && new URL(event.request.url).origin === self.location.origin) {
        const copy = response.clone();
        event.waitUntil(caches.open(CACHE).then((c) => c.put(event.request, copy)));
      }
      return response;
    } catch (err) {
      if (event.request.mode === 'navigate') return caches.match('index.html');
      return new Response('Offline and not cached', { status: 503, headers: { 'Content-Type': 'text/plain' } });
    }
  })());
});
