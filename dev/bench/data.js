window.BENCHMARK_DATA = {
  "lastUpdate": 1783094159283,
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
          "id": "b0cc6df3c2aa559767402d0615256c88f9477599",
          "message": "clump: drop redundant 'in Rust' from README opener (no-marketing-tone rule)",
          "timestamp": "2026-04-26T11:21:28-04:00",
          "tree_id": "ae8cd085a7a67304ff91a45ddf75084e8f9fac86",
          "url": "https://github.com/arclabs561/clump/commit/b0cc6df3c2aa559767402d0615256c88f9477599"
        },
        "date": 1777217880346,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 639999,
            "range": "± 28608",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3341262,
            "range": "± 21109",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 52256935,
            "range": "± 238165",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5311366,
            "range": "± 104219",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 34798065,
            "range": "± 622754",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 282380188,
            "range": "± 2700474",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 525946477,
            "range": "± 3551096",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26969955,
            "range": "± 136693",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4396381850,
            "range": "± 31393568",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5246014,
            "range": "± 12685",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 20873828,
            "range": "± 90905",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 540816236,
            "range": "± 3482712",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 536351276,
            "range": "± 4437293",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1219462869,
            "range": "± 8249042",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2635108,
            "range": "± 11701",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10133608,
            "range": "± 108748",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 39471056,
            "range": "± 199381",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 246196186,
            "range": "± 3214182",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 140486,
            "range": "± 4265",
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
          "id": "aebe1f233eb391181efa33b2b44fc4786f028c21",
          "message": "clump: CI -- nextest, fan-in gate, concurrency cancel, CARGO_INCREMENTAL=0",
          "timestamp": "2026-04-26T12:26:32-04:00",
          "tree_id": "0749288455f8b10b4038134c6db76b0c96000a0e",
          "url": "https://github.com/arclabs561/clump/commit/aebe1f233eb391181efa33b2b44fc4786f028c21"
        },
        "date": 1777221697070,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 487946,
            "range": "± 12001",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 2690521,
            "range": "± 69055",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 42801825,
            "range": "± 625388",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 4393995,
            "range": "± 64400",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 27721111,
            "range": "± 64144",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 230996737,
            "range": "± 230859",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 430435887,
            "range": "± 712675",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 22297026,
            "range": "± 95888",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 3646528343,
            "range": "± 18761378",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 4452920,
            "range": "± 7142",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 17623369,
            "range": "± 21098",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 453723177,
            "range": "± 3389723",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 414003817,
            "range": "± 3093165",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1020394002,
            "range": "± 7170018",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2180577,
            "range": "± 7271",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 8561329,
            "range": "± 29198",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 33453853,
            "range": "± 95135",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 205498742,
            "range": "± 439301",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 114445,
            "range": "± 132",
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
          "id": "7722373e269d2f7bdf7445476d9540b2f243031a",
          "message": "ci: gate publish workflow on cargo-semver-checks",
          "timestamp": "2026-04-28T20:11:57-04:00",
          "tree_id": "eb25da2295bc4d22bf4d0160e93da253b8d767c8",
          "url": "https://github.com/arclabs561/clump/commit/7722373e269d2f7bdf7445476d9540b2f243031a"
        },
        "date": 1777422836415,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 603095,
            "range": "± 18053",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3434719,
            "range": "± 8861",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53592339,
            "range": "± 161361",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5590883,
            "range": "± 24031",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35467611,
            "range": "± 144109",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 289491171,
            "range": "± 680740",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 539034949,
            "range": "± 6700796",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 28250392,
            "range": "± 58009",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4540255763,
            "range": "± 22139638",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5582276,
            "range": "± 59130",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22218053,
            "range": "± 43431",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 567794812,
            "range": "± 5654821",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 542723699,
            "range": "± 2649851",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1275591395,
            "range": "± 5804467",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2799888,
            "range": "± 15221",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10886736,
            "range": "± 19671",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 42439710,
            "range": "± 79645",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 260853677,
            "range": "± 272331",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 144481,
            "range": "± 299",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "cffdf7c7348c85acad994ea084bf31ef69ff6e8c",
          "message": "release: 0.5.7",
          "timestamp": "2026-06-10T10:26:29-04:00",
          "tree_id": "7d55fb4adb0e2f35f342bda2e343783cd4594d74",
          "url": "https://github.com/arclabs561/clump/commit/cffdf7c7348c85acad994ea084bf31ef69ff6e8c"
        },
        "date": 1781102858548,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 578797,
            "range": "± 21011",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3440768,
            "range": "± 15643",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53624576,
            "range": "± 413983",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5515755,
            "range": "± 122733",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35853751,
            "range": "± 95567",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 289854603,
            "range": "± 2495919",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 539283445,
            "range": "± 2900155",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 27866125,
            "range": "± 195066",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4528137548,
            "range": "± 30184872",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5528232,
            "range": "± 14658",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22570941,
            "range": "± 590894",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 563188164,
            "range": "± 5399877",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 538990183,
            "range": "± 3270446",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1268961195,
            "range": "± 10815676",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2755960,
            "range": "± 23403",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10811890,
            "range": "± 309503",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 42414150,
            "range": "± 820301",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 258196296,
            "range": "± 3918506",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 145657,
            "range": "± 3939",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "b5562de48161e77148c3278c308491c632ca2103",
          "message": "ci: gitignore the semver-checks scratch dir; cargo publish refuses dirty trees",
          "timestamp": "2026-06-10T10:45:33-04:00",
          "tree_id": "aa5e4bef50ebb38af72d0e1737f839f9759372e1",
          "url": "https://github.com/arclabs561/clump/commit/b5562de48161e77148c3278c308491c632ca2103"
        },
        "date": 1781103826485,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 579924,
            "range": "± 15388",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3448927,
            "range": "± 75646",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53830472,
            "range": "± 167354",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5508665,
            "range": "± 84930",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 36030684,
            "range": "± 99436",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 290332756,
            "range": "± 1953676",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 544444353,
            "range": "± 5694363",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 27908617,
            "range": "± 502310",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4593105216,
            "range": "± 43568915",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5542695,
            "range": "± 18370",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22385554,
            "range": "± 129919",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 566567423,
            "range": "± 3113436",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 550394494,
            "range": "± 3697959",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1275639729,
            "range": "± 7618973",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2773715,
            "range": "± 10590",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10831010,
            "range": "± 259360",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 42677685,
            "range": "± 292883",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 258928486,
            "range": "± 2498545",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 145493,
            "range": "± 343",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "8510db22cef332e5a30bd8a0b664e7cff5d4f9c4",
          "message": "dep: bump innr to 0.4",
          "timestamp": "2026-06-10T12:06:40-04:00",
          "tree_id": "b6f2533aa449cc8c6a18ebb5534ce02788df7dfd",
          "url": "https://github.com/arclabs561/clump/commit/8510db22cef332e5a30bd8a0b664e7cff5d4f9c4"
        },
        "date": 1781108775877,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 634196,
            "range": "± 5956",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3280270,
            "range": "± 12903",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 52067151,
            "range": "± 1406760",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5249107,
            "range": "± 14092",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 34351417,
            "range": "± 187317",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 281905316,
            "range": "± 1009326",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 524518175,
            "range": "± 2986503",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26544711,
            "range": "± 107983",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4333235581,
            "range": "± 21339127",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5313326,
            "range": "± 22334",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 21416579,
            "range": "± 138990",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 547837198,
            "range": "± 1282291",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 545674503,
            "range": "± 4686872",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1234424892,
            "range": "± 4533461",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2545321,
            "range": "± 22385",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 9786706,
            "range": "± 44342",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 38207876,
            "range": "± 244823",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 235817859,
            "range": "± 776541",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 145413,
            "range": "± 730",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "5df44aea7b112d7daa903367ed8f7d9bada698e9",
          "message": "ci: bump checkout/cache/artifact actions to node24-ready majors",
          "timestamp": "2026-06-10T12:24:25-04:00",
          "tree_id": "a61a3c55e4b2d1112aa27cc20fca01b844d3d1ed",
          "url": "https://github.com/arclabs561/clump/commit/5df44aea7b112d7daa903367ed8f7d9bada698e9"
        },
        "date": 1781110024305,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 636615,
            "range": "± 5510",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3299485,
            "range": "± 81718",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 51989497,
            "range": "± 101932",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5253777,
            "range": "± 9873",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 34669150,
            "range": "± 162977",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 281264388,
            "range": "± 2330592",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 522552776,
            "range": "± 791779",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26556531,
            "range": "± 175202",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4320494337,
            "range": "± 7193373",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5303301,
            "range": "± 8847",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 21105591,
            "range": "± 85352",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 547462127,
            "range": "± 1389208",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 536309776,
            "range": "± 2869075",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1234295819,
            "range": "± 1374073",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2543737,
            "range": "± 31524",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 9782153,
            "range": "± 34221",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 37986071,
            "range": "± 194475",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 236932691,
            "range": "± 1041835",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 143165,
            "range": "± 932",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "2bad89a9497979c8e7bee63e5535a4f3f172c062",
          "message": "docs: bump README install pin to 0.5.7",
          "timestamp": "2026-06-12T22:52:07-04:00",
          "tree_id": "666cd01884ebdfe13d712073d173290376119d9d",
          "url": "https://github.com/arclabs561/clump/commit/2bad89a9497979c8e7bee63e5535a4f3f172c062"
        },
        "date": 1781320124968,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 636245,
            "range": "± 4047",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3280206,
            "range": "± 7118",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 51891253,
            "range": "± 139132",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5246208,
            "range": "± 12885",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 34512311,
            "range": "± 97740",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 279879978,
            "range": "± 455846",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 520753786,
            "range": "± 670272",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26504518,
            "range": "± 187845",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4311812873,
            "range": "± 20507730",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5235411,
            "range": "± 8439",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 20855204,
            "range": "± 50041",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 539746065,
            "range": "± 2652306",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 533138525,
            "range": "± 1984483",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1215680897,
            "range": "± 4430480",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2516471,
            "range": "± 64307",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 9610928,
            "range": "± 35798",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 37447990,
            "range": "± 835818",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 232762816,
            "range": "± 1998047",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 141393,
            "range": "± 488",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "971b858d25973c55733ab6191dfe7d39af212305",
          "message": "docs: bump README install pin to 0.5.7",
          "timestamp": "2026-06-12T22:52:07-04:00",
          "tree_id": "666cd01884ebdfe13d712073d173290376119d9d",
          "url": "https://github.com/arclabs561/clump/commit/971b858d25973c55733ab6191dfe7d39af212305"
        },
        "date": 1781994997653,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 481789,
            "range": "± 9087",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 2680170,
            "range": "± 13724",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 42792112,
            "range": "± 324868",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 4367958,
            "range": "± 21658",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 27707244,
            "range": "± 501001",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 230971351,
            "range": "± 1674891",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 430230509,
            "range": "± 3377632",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 22145861,
            "range": "± 28053",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 3648214991,
            "range": "± 32142419",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 4435743,
            "range": "± 96184",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 17720651,
            "range": "± 325744",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 453717875,
            "range": "± 3551212",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 426524373,
            "range": "± 3065981",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1022292420,
            "range": "± 6189284",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2162310,
            "range": "± 9810",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 8545114,
            "range": "± 32104",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 33863282,
            "range": "± 920779",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 205212577,
            "range": "± 2230543",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 115490,
            "range": "± 1296",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "2af76364334328c73b9be3cc6db9e9a7f6c4278b",
          "message": "clump: run example assertions in CI\n\nThe asserting examples compiled but never ran in CI; added a run step so a\nregression that breaks an example fails CI instead of passing silently.",
          "timestamp": "2026-06-25T16:51:00-04:00",
          "tree_id": "3cfa0ead2cd72a9f721c31d497a849ddd6ab1000",
          "url": "https://github.com/arclabs561/clump/commit/2af76364334328c73b9be3cc6db9e9a7f6c4278b"
        },
        "date": 1782421744100,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 576476,
            "range": "± 17350",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3465584,
            "range": "± 9680",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53611577,
            "range": "± 456589",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5505328,
            "range": "± 70181",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35928887,
            "range": "± 71506",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 289870114,
            "range": "± 1615094",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 539216928,
            "range": "± 1537657",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 27830926,
            "range": "± 43585",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4493833612,
            "range": "± 27113374",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5538223,
            "range": "± 11947",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22024019,
            "range": "± 20928",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 564522941,
            "range": "± 4218330",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 535004315,
            "range": "± 2041489",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1265946891,
            "range": "± 10497606",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2616563,
            "range": "± 21379",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10271356,
            "range": "± 46803",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 39938295,
            "range": "± 274693",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 244904383,
            "range": "± 2151344",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 144599,
            "range": "± 525",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "e735fa0808ee05f3f21e18f10d617f52c532a55c",
          "message": "examples: exit 0 on missing dataset and satisfy fmt/clippy\n\nData-gated examples now no-op (exit 0) when the dataset is absent instead of returning failure, so CI that runs examples passes; plus fmt + clippy fixes.",
          "timestamp": "2026-06-25T19:32:53-04:00",
          "tree_id": "db0c26ad61058908f9e4a70addc4eeb9d248437e",
          "url": "https://github.com/arclabs561/clump/commit/e735fa0808ee05f3f21e18f10d617f52c532a55c"
        },
        "date": 1782431437331,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 638482,
            "range": "± 1559",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3282462,
            "range": "± 12880",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 52028651,
            "range": "± 860300",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5295193,
            "range": "± 33326",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 34555594,
            "range": "± 127265",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 281251789,
            "range": "± 580815",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 522589688,
            "range": "± 1010737",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26620085,
            "range": "± 124919",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4324398524,
            "range": "± 19220515",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5297552,
            "range": "± 9649",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 21349760,
            "range": "± 120933",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 547477712,
            "range": "± 867697",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 532975407,
            "range": "± 2858959",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1234644861,
            "range": "± 2676780",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2512829,
            "range": "± 4484",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 9665100,
            "range": "± 21269",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 37550181,
            "range": "± 95198",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 232841960,
            "range": "± 447625",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 141415,
            "range": "± 4185",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "4899372d3d390fb4554be9287c3c532981b746e9",
          "message": "clump: add rosetta correctness fixtures vs scikit-learn\n\nWave 4 of the Rosetta cross-language correctness program, completing the planned\ncrate priority. First crate in the QUALITY tolerance class (clustering output is\na partition, not a value).\n\nDBSCAN is deterministic given (eps, min_samples), so clump's labels must match\nsklearn's exactly up to a label permutation (ARI = 1.0). The test does not\ncompare label values; it asserts the two labelings induce the same partition:\nidentical noise set, and the co-cluster relation agrees on every pair. Three\nwell-separated blobs plus isolated noise make the result independent of boundary\nconventions (self-counting in min_samples, < vs <= on eps).\n\nk-means uses different k-means++ RNG in clump (f32) vs sklearn (f64), so the\ncentroids are not identical; inertia (WCSS) is compared within 2%, not exactly.\nOn separable blobs both reach the same optimum, so the gap is small.\n\nserde/serde_json are dev-dependencies only.",
          "timestamp": "2026-06-27T13:50:55-04:00",
          "tree_id": "a5e7c873cb16917a2ff30199bdafcdd4ce333bd6",
          "url": "https://github.com/arclabs561/clump/commit/4899372d3d390fb4554be9287c3c532981b746e9"
        },
        "date": 1782583728740,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 596204,
            "range": "± 10297",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3431799,
            "range": "± 74318",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53620780,
            "range": "± 161597",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5530210,
            "range": "± 23792",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 35438983,
            "range": "± 113507",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 289589110,
            "range": "± 2970053",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 539051958,
            "range": "± 2357771",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 27948481,
            "range": "± 174766",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4559660065,
            "range": "± 28561001",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5601011,
            "range": "± 8701",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22491755,
            "range": "± 216191",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 570635802,
            "range": "± 5280852",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 546949710,
            "range": "± 5701823",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1285785342,
            "range": "± 8152604",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2642109,
            "range": "± 19626",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10349572,
            "range": "± 47804",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 40865747,
            "range": "± 490673",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 246641880,
            "range": "± 1703430",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 145369,
            "range": "± 304",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "b3d5c73525a495babea64bc37433e0a4b0693f15",
          "message": "clump: add examples gallery (examples/README.md)\n\nQuestion-driven gallery over the six examples with real captured output, per the\nsklearn-style gold standard: each entry states the question it answers, the run\ncommand, and verbatim sample output; the mnist example is marked data-gated with\nits fetch script. README Examples section now links the gallery.",
          "timestamp": "2026-06-29T21:09:56-04:00",
          "tree_id": "fc2999d47a7609d8844b271f7ddb8665885dba74",
          "url": "https://github.com/arclabs561/clump/commit/b3d5c73525a495babea64bc37433e0a4b0693f15"
        },
        "date": 1782782807913,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 637220,
            "range": "± 2103",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3289630,
            "range": "± 99941",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 51927939,
            "range": "± 124997",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5247616,
            "range": "± 11708",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 34342142,
            "range": "± 302750",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 281398768,
            "range": "± 906254",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 522675020,
            "range": "± 3262596",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 26523980,
            "range": "± 61860",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4339604252,
            "range": "± 21044165",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5255699,
            "range": "± 16589",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 21030227,
            "range": "± 205741",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 543082749,
            "range": "± 2849871",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 557333286,
            "range": "± 5213288",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1223944224,
            "range": "± 4279978",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2545666,
            "range": "± 65490",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 9776471,
            "range": "± 35335",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 38059447,
            "range": "± 1268823",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 235242998,
            "range": "± 456293",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 143585,
            "range": "± 650",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "committer": {
            "email": "henry@henrywallace.io",
            "name": "Henry Wallace",
            "username": "arclabs561"
          },
          "distinct": true,
          "id": "9f3f1a824fa2d95ea0566c7e68f04de4a82798c8",
          "message": "hdbscan: select clusters in post-order, not reverse id order\n\nReverse-id selection assumed cluster ids are topological, but a genuine\nsplit allocates the parent id BEFORE fresh child ids, so a leaf born at\nan early merge can sit below a later ancestor. Such leaves were visited\nafter their ancestor had selected itself and deselected the subtree,\nand the unconditional leaf-select re-selected them: on a three-blob\ncontrol the reference implementation returns 3 clusters, clump returned\n6 (two blobs split into their sub-structure). Explicit post-order fixes\nboth that re-selection and the latent sibling hazard (a parent reading a\nlower-id internal child's unpropagated subtree stability).\n\nFound by pairing a discovery agent's structural claim with an empirical\noracle: a fixture generated from the McInnes hdbscan package\n(separated-blobs control + 3-level nested-density case). The agent's\nsecond claim, a core-distance off-by-one, was REFUTED by the same\noracle: shifting the convention one neighbor nearer worsened agreement\n(ARI 0.74 -> 0.46), and with the traversal fixed the shipped convention\nmatches the reference exactly (ARI 1.0 on both cases). The new rosetta\ntest asserts partition equality (noise set + pairwise co-cluster\nrelation) per the existing rosetta_clump.rs pattern and runs in the\ndefault gate.",
          "timestamp": "2026-07-03T11:37:48-04:00",
          "tree_id": "d44fbf1b42073f91d075ddae746417fa0221924f",
          "url": "https://github.com/arclabs561/clump/commit/9f3f1a824fa2d95ea0566c7e68f04de4a82798c8"
        },
        "date": 1783094158756,
        "tool": "cargo",
        "benches": [
          {
            "name": "kmeans/n1000_d16_k10",
            "value": 598311,
            "range": "± 12098",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d16_k10",
            "value": 3438213,
            "range": "± 57785",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n10000_d16_k100",
            "value": 53581736,
            "range": "± 616943",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n1000_d128_k10",
            "value": 5558579,
            "range": "± 19271",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k10",
            "value": 36905722,
            "range": "± 125082",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n50000_d16_k100",
            "value": 289457425,
            "range": "± 359381",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n100000_d16_k100",
            "value": 539048470,
            "range": "± 3870583",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n5000_d128_k10_highmag",
            "value": 28143353,
            "range": "± 34302",
            "unit": "ns/iter"
          },
          {
            "name": "kmeans/n200000_d128_k50",
            "value": 4542554941,
            "range": "± 20564210",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n1000_d16",
            "value": 5582653,
            "range": "± 26104",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n2000_d16",
            "value": 22194802,
            "range": "± 179794",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n10000_d16",
            "value": 568468341,
            "range": "± 5987651",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n50000_d3",
            "value": 531215827,
            "range": "± 4015560",
            "unit": "ns/iter"
          },
          {
            "name": "dbscan/n15000_d16",
            "value": 1276350432,
            "range": "± 4550581",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n500_d16",
            "value": 2670587,
            "range": "± 12939",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n1000_d16",
            "value": 10351619,
            "range": "± 84028",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n2000_d16",
            "value": 40406745,
            "range": "± 748395",
            "unit": "ns/iter"
          },
          {
            "name": "hdbscan/n5000_d16",
            "value": 245385779,
            "range": "± 2812896",
            "unit": "ns/iter"
          },
          {
            "name": "minibatch_kmeans/5x200_d16_k10",
            "value": 144308,
            "range": "± 248",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}