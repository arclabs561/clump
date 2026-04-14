window.BENCHMARK_DATA = {
  "lastUpdate": 1776173046084,
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
      }
    ]
  }
}