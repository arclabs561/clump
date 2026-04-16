window.BENCHMARK_DATA = {
  "lastUpdate": 1776353397469,
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
      }
    ]
  }
}