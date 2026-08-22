# Baseline preprocessing statique

- Symptôme: le run de production stable `smi-prepare-pilot --workers 1` n'utilise qu'environ un cœur CPU.
- Débit observé sans benchmark dédié: environ 350–370 assistant-target tokens/s sur les premiers 607583 tokens confirmés.
- Entrée: cache local immuable 62/62, longueur maximale 2048, shuffle buffer 10000, quotas totalisant 10M tokens.
- Référence connue: aucune implémentation longue durée plus rapide et stable. Le benchmark court à 16 threads n'est pas une référence valide car il s'est ensuite bloqué.
- Blast radius: audit read-only du code; aucun benchmark, aucune interruption ou mutation du corpus actif.
- Arrêt: goulots classés avec références de code, mécanismes, gains plausibles et plan sûr.
