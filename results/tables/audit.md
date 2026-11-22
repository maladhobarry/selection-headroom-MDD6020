| check | finding | action |
|---|---|---|
| Clinical rows / patients | 754 / 732 | Single-embryo transfers kept |
| Double-embryo transfers | 22 patients | Excluded (one outcome for two embryos) |
| Image names not in NNNN_NN.png form | 545 of 754 | Numeric (patient, embryo) key for all joins |
| Expert-grade file | 2344 rows, 2 rows in conflicting duplicates | First occurrence kept |
| Transferred images without expert grade | 846_01.png | Excluded |
| EXP: clinical file vs expert-grade file | 96.5% identical | Expert-grade file used; clinical grades in sensitivity S4 |
| ICM: clinical file vs expert-grade file | 97.5% identical | Expert-grade file used; clinical grades in sensitivity S4 |
| TE: clinical file vs expert-grade file | 97.1% identical | Expert-grade file used; clinical grades in sensitivity S4 |
| AMH parsing rules | month_year=377; missing=165; day_month=124; numeric=88 | Reversed date conversion / first number of sums; flagged |
| Endo parsing rules | numeric=598; day_month=144; missing=12 | Reversed date conversion / first number of sums; flagged |
| COC parsing rules | numeric=741; sum_first=8; missing=5 | Reversed date conversion / first number of sums; flagged |
| MII parsing rules | numeric=753; sum_first=1 | Reversed date conversion / first number of sums; flagged |
| Variable 'Fond' | Not documented in Kromp et al. 2023 (Table 1) | Not used |
| Mean expansion, day 4: siblings vs transferred | 1.97 vs 1.46 | Day 4 excluded from strategy comparison (siblings likely photographed later) |
| Mean expansion, day 5: siblings vs transferred | 2.35 vs 2.74 | Day 4 excluded from strategy comparison (siblings likely photographed later) |
| Annotator files | 12 files, 75 incomplete annotations; consensus on 300 images | Incomplete criteria ignored per criterion |
