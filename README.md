# Mario AI with Genetic Algorithm

## 作品概要
スーパーマリオブラザーズ（を模したゲーム）を自動操作するAIを、遺伝的アルゴリズムを用いて制作しました。

参考サイト: [Morishin Blog](https://blog.morishin.me/posts/2016/12/19/marioai-2009)

---

## 使用言語
- Python

---

## 工夫点

### 1. トーナメント選択法の実装

- トーナメント選択法 (Tournament Selection) により、強い個体（エージェント）を選択する際の多様性を確保。
- 5つのランダムな個体から最も高いスコアを持つ個体を選び、次世代に継承します。

```python
def tournament_selection(pop, k=5):
    selected = []
    for _ in range(len(pop)):
        aspirants = [pop[i] for i in numpy.random.randint(len(pop), size=k)]
        selected.append(max(aspirants, key=lambda ind: ind.reward))
    return selected
```

- `k=5` により、選択の際の多様性を確保しつつ、局所解に陥るのを防止しています。

### 2. エリート選択の導入

- 各世代で最も優れた個体（エリート）を次世代に必ず引き継ぐことで、進化が逆行しないようにしています。

```python
numberOfElites = 5
sorted_rewards = sorted(rewards, key=lambda individual_reward: individual_reward.reward, reverse=True)
elite_individuals = list(map(lambda e: e.individual, sorted_rewards[:numberOfElites]))
```

- `numberOfElites = 5` とすることで、過度なエリート化を防ぎつつ、確実に良い個体を次世代に残します。

### 3. 動的な突然変異率の設定

- 突然変異率（mutation rate）を、世代数に応じて徐々に減少させる設計。
- これにより、初期は多様性を確保し、後半は収束を促進します。

```python
mutation_rate = max(0.1, 0.3 - len(next_individuals) * 0.01)
Controller.mutate(next_individuals, mutation_rate=mutation_rate)
```

- `max(0.1, 0.3 - len(next_individuals) * 0.01)` で、最小0.1の突然変異率を確保し、進化が停滞しないようにしています。


---

## 今後の改善点
- AIの行動評価関数の見直し（報酬設計の最適化）
- 進化戦略 (Evolution Strategy) やニューラルネットワークの導入によるパフォーマンス向上
- マルチプロセッシングを利用した計算時間の短縮

