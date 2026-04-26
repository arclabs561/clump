window.BENCHMARK_DATA = {
  "lastUpdate": 1777171213507,
  "repoUrl": "https://github.com/arclabs561/clump",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "7707ff3a109d3409d54fa79cbd7203a44523b48e",
          "message": "fix bench CI: add contents write permission for gh-pages push",
          "timestamp": "2026-04-14T09:04:59-04:00",
          "tree_id": "1d46568c932953b6c76ca01ce94b0b78466268d5",
          "url": "https://github.com/arclabs561/clump/commit/7707ff3a109d3409d54fa79cbd7203a44523b48e"
        },
        "date": 1776173045501,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 657626,
            "range": "± 10149",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3371974,
            "range": "± 22837",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53103130,
            "range": "± 218851",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5293932,
            "range": "± 13352",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35122335,
            "range": "± 123153",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 286402334,
            "range": "± 333193",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 533671732,
            "range": "± 1529390",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26854848,
            "range": "± 61430",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4326517420,
            "range": "± 11738244",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5295519,
            "range": "± 10961",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 21090282,
            "range": "± 45760",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 575930900,
            "range": "± 1154203",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 521055961,
            "range": "± 1212854",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1294822925,
            "range": "± 1460207",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 3538532,
            "range": "± 55366",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 13679357,
            "range": "± 355727",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 53738420,
            "range": "± 1724346",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 349045986,
            "range": "± 4259300",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 143646,
            "range": "± 1141",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "29b7dc5c113d3046d8e09c8967d1ce0f16908d58",
          "message": "add cargo-semver-checks to CI",
          "timestamp": "2026-04-14T09:22:58-04:00",
          "tree_id": "2676d9e4fd1e3cff54956a608fc4d2bd285583dc",
          "url": "https://github.com/arclabs561/clump/commit/29b7dc5c113d3046d8e09c8967d1ce0f16908d58"
        },
        "date": 1776174056159,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 659858,
            "range": "± 1991",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3370870,
            "range": "± 13013",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53292538,
            "range": "± 766088",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5365356,
            "range": "± 17786",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35179765,
            "range": "± 157050",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 287666508,
            "range": "± 1424495",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 536577279,
            "range": "± 1701856",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 27396725,
            "range": "± 208773",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4367730562,
            "range": "± 20267954",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5633418,
            "range": "± 19700",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22910956,
            "range": "± 155653",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 606663899,
            "range": "± 3624432",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 527759940,
            "range": "± 3747721",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1368469875,
            "range": "± 4116329",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2911157,
            "range": "± 7015",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 11198956,
            "range": "± 43868",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 44004567,
            "range": "± 296425",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 277765011,
            "range": "± 709739",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 143231,
            "range": "± 912",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "3014728ceadb3b189841536413671861abbec84f",
          "message": "clump: bump 0.5.4 (DBSCAN grid optimization, benchmarks, MSRV fixes)",
          "timestamp": "2026-04-14T17:50:26-04:00",
          "tree_id": "cbdaab002ec699d6af5639b965ccb78075f21295",
          "url": "https://github.com/arclabs561/clump/commit/3014728ceadb3b189841536413671861abbec84f"
        },
        "date": 1776208696148,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 631762,
            "range": "± 12164",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3507669,
            "range": "± 38844",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 54812567,
            "range": "± 371541",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5502289,
            "range": "± 19390",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 36314312,
            "range": "± 439237",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 295852229,
            "range": "± 446460",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 552746931,
            "range": "± 3530474",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 27875547,
            "range": "± 189478",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4510460391,
            "range": "± 33971807",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5551520,
            "range": "± 17458",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22104754,
            "range": "± 61940",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 564776668,
            "range": "± 3241474",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 556105350,
            "range": "± 5217102",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1270356999,
            "range": "± 7912759",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 3063694,
            "range": "± 39048",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 11878221,
            "range": "± 51678",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 46768081,
            "range": "± 232446",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 284755911,
            "range": "± 2978475",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 146913,
            "range": "± 192",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "ba6182e452cf06b9244eb29d72ec961b2bf8e8c4",
          "message": "readme: lead with HDBSCAN differentiator over linfa-clustering",
          "timestamp": "2026-04-15T20:59:14-04:00",
          "tree_id": "a15189dbb1715afaa50f8ef738368dfe242586db",
          "url": "https://github.com/arclabs561/clump/commit/ba6182e452cf06b9244eb29d72ec961b2bf8e8c4"
        },
        "date": 1776302183619,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 618935,
            "range": "± 9459",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3543696,
            "range": "± 68575",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 54984376,
            "range": "± 1479412",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5548017,
            "range": "± 22688",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 36584244,
            "range": "± 87076",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 296645831,
            "range": "± 548916",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 554961077,
            "range": "± 2556879",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 28254098,
            "range": "± 51489",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4532139966,
            "range": "± 33076689",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5544965,
            "range": "± 10757",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22188407,
            "range": "± 86150",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 563590677,
            "range": "± 1821787",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 543472374,
            "range": "± 2440441",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1267545450,
            "range": "± 5459001",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 3049168,
            "range": "± 12302",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 11875712,
            "range": "± 41752",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 46286753,
            "range": "± 771177",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 284080777,
            "range": "± 2917279",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 144958,
            "range": "± 348",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "e8fd57e5b383fa70e48ab86d2282a875c077f038",
          "message": "prep: add rkhs feature gate for future CLAM integration",
          "timestamp": "2026-04-15T21:08:54-04:00",
          "tree_id": "565d5d0098bfb67d6af750a755338ec041541bba",
          "url": "https://github.com/arclabs561/clump/commit/e8fd57e5b383fa70e48ab86d2282a875c077f038"
        },
        "date": 1776302758073,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 610245,
            "range": "± 20773",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3548336,
            "range": "± 24514",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 54823365,
            "range": "± 299023",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5547834,
            "range": "± 10386",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 37218816,
            "range": "± 612644",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 296604998,
            "range": "± 2568347",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 553166647,
            "range": "± 6189407",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 28083623,
            "range": "± 384749",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4541169838,
            "range": "± 26852121",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5545975,
            "range": "± 117712",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22024662,
            "range": "± 64061",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 565652348,
            "range": "± 4060455",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 541818646,
            "range": "± 3376930",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1268702851,
            "range": "± 6469504",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 3040808,
            "range": "± 18237",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 11909385,
            "range": "± 279060",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 46417223,
            "range": "± 1098018",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 281424893,
            "range": "± 2356572",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 145231,
            "range": "± 198",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "6dc3d9477cd71b367a85f07b9752cd0d99121554",
          "message": "hdbscan: reduce allocations in mutual reachability computation",
          "timestamp": "2026-04-16T10:03:20-04:00",
          "tree_id": "92e8376c4e91f57b79e4efdf6fb8205d7e613ff0",
          "url": "https://github.com/arclabs561/clump/commit/6dc3d9477cd71b367a85f07b9752cd0d99121554"
        },
        "date": 1776349220586,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 634576,
            "range": "± 2369",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3288866,
            "range": "± 15028",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 51862900,
            "range": "± 717097",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5252563,
            "range": "± 64380",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35069531,
            "range": "± 234977",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 279709743,
            "range": "± 2605781",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 521814945,
            "range": "± 2700179",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26803797,
            "range": "± 65020",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4331484103,
            "range": "± 45551367",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5297585,
            "range": "± 12252",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 21133454,
            "range": "± 72131",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 549658134,
            "range": "± 4480071",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 523311153,
            "range": "± 1323895",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1241379270,
            "range": "± 7367124",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2511210,
            "range": "± 20676",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 9659513,
            "range": "± 17074",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 37505223,
            "range": "± 239505",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 234422408,
            "range": "± 1567007",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 142379,
            "range": "± 746",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "7ed59ace840235d46063e3fcbe524ff182eaa489",
          "message": "clump: migrate clam feature from rkhs to hopfield\n\nCLAM clustering now depends directly on hopfield for AM energy functions\ninstead of pulling in the full rkhs crate.",
          "timestamp": "2026-04-16T11:15:39-04:00",
          "tree_id": "a235dab332352d1144c84a7d4bdb9bca801ac21b",
          "url": "https://github.com/arclabs561/clump/commit/7ed59ace840235d46063e3fcbe524ff182eaa489"
        },
        "date": 1776353396491,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 458518,
            "range": "± 7220",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 2690622,
            "range": "± 60243",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 42904717,
            "range": "± 102167",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 4328007,
            "range": "± 62224",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 28025396,
            "range": "± 484369",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 232302545,
            "range": "± 1204293",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 433488868,
            "range": "± 2172105",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 22129503,
            "range": "± 36904",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 3682907638,
            "range": "± 25955012",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 4447516,
            "range": "± 100139",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 17845756,
            "range": "± 154687",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 455596894,
            "range": "± 4095658",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 420624657,
            "range": "± 2515906",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1021783686,
            "range": "± 5452749",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2094947,
            "range": "± 7253",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 8170092,
            "range": "± 170532",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 32581322,
            "range": "± 1082890",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 195643554,
            "range": "± 1865620",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 114912,
            "range": "± 295",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "7c88b889a924875bd01e515b88775d924eaca283",
          "message": "clump: fix stale version, remove comparison claim, add examples table",
          "timestamp": "2026-04-16T20:52:09-04:00",
          "tree_id": "c60e6a389d3740f13b0edd16435a5ffd0c3ba0e1",
          "url": "https://github.com/arclabs561/clump/commit/7c88b889a924875bd01e515b88775d924eaca283"
        },
        "date": 1776388333974,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 651632,
            "range": "± 8155",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3609597,
            "range": "± 58712",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 51982918,
            "range": "± 639232",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 6000971,
            "range": "± 34531",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 37399041,
            "range": "± 244668",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 281646604,
            "range": "± 831721",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 529725459,
            "range": "± 2102835",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 29795566,
            "range": "± 257706",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4931669306,
            "range": "± 21820683",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5431462,
            "range": "± 6745",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 21696724,
            "range": "± 87960",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 595134469,
            "range": "± 4001632",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 483758407,
            "range": "± 1618832",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1356548940,
            "range": "± 2143958",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2538038,
            "range": "± 6752",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 9820772,
            "range": "± 91115",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 38181721,
            "range": "± 768047",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 242889804,
            "range": "± 1154220",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 144140,
            "range": "± 321",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "586c14042767339aaad98f79bd70f009e79d2314",
          "message": "clump: 0.5.6 — CLAM module (clam feature via hopfield), HDBSCAN alloc reduction, clippy fixes",
          "timestamp": "2026-04-20T19:19:20-04:00",
          "tree_id": "3e18023ee655995bc4c3d0e77162a04ca0026335",
          "url": "https://github.com/arclabs561/clump/commit/586c14042767339aaad98f79bd70f009e79d2314"
        },
        "date": 1776728151132,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 638642,
            "range": "± 7467",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3303748,
            "range": "± 13328",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 51975413,
            "range": "± 798399",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5281829,
            "range": "± 79001",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 34381350,
            "range": "± 123735",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 280719723,
            "range": "± 1490839",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 522358516,
            "range": "± 3286554",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26947488,
            "range": "± 55073",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4325837830,
            "range": "± 18966160",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5245768,
            "range": "± 10888",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 20960371,
            "range": "± 135241",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 542577708,
            "range": "± 2992191",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 535849425,
            "range": "± 3561517",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1215404189,
            "range": "± 2896031",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2530954,
            "range": "± 7333",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 9831605,
            "range": "± 68432",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 37867366,
            "range": "± 116204",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 235362419,
            "range": "± 562610",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 141790,
            "range": "± 1856",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "17167b62deee438d91b6d5d205db7bb470249686",
          "message": "ci: use --all-features instead of explicit feature list in clippy",
          "timestamp": "2026-04-20T21:25:53-04:00",
          "tree_id": "f8af8503f8b392a860a7c0496652be1b0f061cce",
          "url": "https://github.com/arclabs561/clump/commit/17167b62deee438d91b6d5d205db7bb470249686"
        },
        "date": 1776735730838,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 638508,
            "range": "± 6013",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3311161,
            "range": "± 47699",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 51899776,
            "range": "± 507636",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5310269,
            "range": "± 18647",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 34779144,
            "range": "± 140401",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 280674391,
            "range": "± 488933",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 521828406,
            "range": "± 558894",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26899079,
            "range": "± 50158",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4314592649,
            "range": "± 17746942",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5240221,
            "range": "± 34665",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 20865930,
            "range": "± 76643",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 538672323,
            "range": "± 5168810",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 529171806,
            "range": "± 1846995",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1214340939,
            "range": "± 3848638",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2525596,
            "range": "± 6370",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 9711410,
            "range": "± 35156",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 37877618,
            "range": "± 195403",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 235395010,
            "range": "± 654689",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 142512,
            "range": "± 589",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "8d51a314c9017a609366a17604269efc65bcc813",
          "message": "clump: README opener -- name the scope concretely",
          "timestamp": "2026-04-25T16:14:11-04:00",
          "tree_id": "30f14600c6718c201835967312c3f5da6a45a6b2",
          "url": "https://github.com/arclabs561/clump/commit/8d51a314c9017a609366a17604269efc65bcc813"
        },
        "date": 1777149300927,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 591752,
            "range": "± 12898",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3443755,
            "range": "± 8979",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53789728,
            "range": "± 295963",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5571768,
            "range": "± 11575",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35483000,
            "range": "± 1122007",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 289593472,
            "range": "± 1348564",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 540067866,
            "range": "± 3648676",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 28165556,
            "range": "± 41842",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4538152107,
            "range": "± 37799527",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5586956,
            "range": "± 8198",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22218747,
            "range": "± 48114",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 569335763,
            "range": "± 1991318",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 532630167,
            "range": "± 7133827",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1275104387,
            "range": "± 6123864",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2788929,
            "range": "± 67932",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10891358,
            "range": "± 18172",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 42412987,
            "range": "± 89364",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 260416675,
            "range": "± 499477",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 144728,
            "range": "± 234",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "0deaf37e9619884c07a48ea3d994c11e67417be8",
          "message": "clump: gate gpu mod on target_os=macos to fix --all-features clippy on Linux CI",
          "timestamp": "2026-04-25T16:41:47-04:00",
          "tree_id": "ad86709b19415312a8ac015e5377ad46a0334a43",
          "url": "https://github.com/arclabs561/clump/commit/0deaf37e9619884c07a48ea3d994c11e67417be8"
        },
        "date": 1777150731088,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 621807,
            "range": "± 15484",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3437896,
            "range": "± 21362",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53601188,
            "range": "± 466707",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5568208,
            "range": "± 25492",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35725142,
            "range": "± 125226",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 289999565,
            "range": "± 2160684",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 539337057,
            "range": "± 2603370",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 28212626,
            "range": "± 312957",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4538323559,
            "range": "± 15588492",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5584474,
            "range": "± 8283",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22179634,
            "range": "± 703100",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 567847730,
            "range": "± 2634571",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 546217141,
            "range": "± 5885629",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1276377015,
            "range": "± 6710574",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2774341,
            "range": "± 14430",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10903896,
            "range": "± 28205",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 42412012,
            "range": "± 297624",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 260566778,
            "range": "± 2760277",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 145027,
            "range": "± 179",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry.wallace@salesforce.com",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "e67a3401bf1278bbc3e3cd9d0c7cab56c54be259",
          "message": "clump: flesh out CONTRIBUTING.md (setup, style, testing, PR expectations)",
          "timestamp": "2026-04-25T22:23:00-04:00",
          "tree_id": "147c1987ed10915b81aa66bdc4aa632a598c2f8a",
          "url": "https://github.com/arclabs561/clump/commit/e67a3401bf1278bbc3e3cd9d0c7cab56c54be259"
        },
        "date": 1777171212957,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 624807,
            "range": "± 22945",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3439916,
            "range": "± 62140",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53680478,
            "range": "± 658631",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5579276,
            "range": "± 11580",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35757853,
            "range": "± 69092",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 290398194,
            "range": "± 2826014",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 543415383,
            "range": "± 2087822",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 28197787,
            "range": "± 37556",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4594202486,
            "range": "± 21757380",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5600153,
            "range": "± 7514",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22391241,
            "range": "± 231397",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 568510887,
            "range": "± 502045",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 544495486,
            "range": "± 8216552",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1276836285,
            "range": "± 15895292",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2781662,
            "range": "± 9118",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10921988,
            "range": "± 23649",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 42510971,
            "range": "± 217329",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 260488543,
            "range": "± 3033841",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 145922,
            "range": "± 1140",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}