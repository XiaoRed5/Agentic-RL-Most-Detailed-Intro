# 共享写作契约 (所有章节 agent 必读)

你在为一个真实代码项目撰写一篇**学术论文风格**的 HTML 报告。四个 agent 各写一章，最后拼成一个 single-file HTML。你只输出**你负责那一章的 HTML 片段**(从 `<h2>` 开始)，不要写 `<html>/<head>/<style>/<body>`,不要重复别人的章节。

## 参考论文风格 (arxiv 2602.11351v1 — BAO)
标题:《Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization》。这是本项目对齐的核心论文之一。ICML/PMLR 双栏学术风:衬线体、白底黑字、编号章节 (1 Introduction / 2 ... )、boxed abstract、booktabs 三线表 (顶/中/底粗线,无竖线)、图带 "Figure N:" 粗体标签的说明文字、行内引用如 (Qian et al., 2025)、方程右侧编号 (1)(2)。**模仿这种严肃、克制、信息密度高的学术语气**,不要营销腔,不要 emoji 堆砌 (正文一律不用 emoji)。中文正文 + 英文术语混排。

## 可用 CSS class (只能用这些,已在 shell 里定义好)
- 章节: `<h2>`, `<h3>`, `<h4>`(斜体小标题)
- 段落: `<p>`, `<p class="lead">`(首段稍大)
- 引用标记: `<span class="cite">(Author et al., 2025)</span>`
- 行内代码: `<code>...</code>`；行内公式 `$...$`；块公式 `$$...$$`(KaTeX 已装好)
- 带编号方程:
  ```
  <div class="eqn-num"><div class="body">$$ ... $$</div><span class="tag">(1)</span></div>
  ```
- 列表: `<ul>`/`<ol>`/`<li>`；紧凑 `<ul class="tight">`
- 图: 
  ```
  <figure><div class="figbox"> ...(inline SVG,白底,学术配色)... </div>
  <figcaption><span class="lbl">Figure N:</span> 说明文字。</figcaption></figure>
  ```
- 三线表:
  ```
  <div class="tabwrap"><table class="aca">
  <caption><span class="lbl">Table N:</span> 说明。</caption>
  <thead><tr><th>...</th><th>...</th></tr></thead>
  <tbody><tr><td>...</td><td class="num">0.59</td></tr>...</tbody>
  </table></div>
  ```
  数字列加 `class="num"`；最优值加 `class="best"`。
- 算法框:
  ```
  <div class="algobox"><div class="cap"><span class="lbl">Algorithm N:</span> 名称</div>
  <div class="body"><span class="kw">for</span> ... <span class="cm"># 注释</span><br>
  <span class="ind">缩进一层</span><br></div></div>
  ```
- 提示框: `<div class="callout"><div class="k">标题</div><p>...</p></div>`
  变体: `callout finding`(绿,正面发现) / `callout limit`(琥珀,诚实边界)
- 代码块: `<pre class="code">...</pre>`(内可用 `<span class="cm/g/r">`)

## 图表编号总规划 (跨章节唯一,不要冲突)
- **Figure 1** = 系统总架构图 (第2章)
- **Figure 2** = 一个训练 step 的数据流管线 (第2章)
- **Figure 3** = 信用分配三打法对比示意 / R2G 折扣前传图 (第3章)
- **Figure 4** = InfoPO 反事实信息增益机制图 (第3章)
- **Table 1** = survey 各层 ↔ 代码模块 ↔ 验证锚点 映射表 (第2章)
- **Table 2** = DAPO 四件套开关 + 落点表 (第4章)
- **Table 3** = advantage estimator 对比 (第4章,可选)
- **Figure 5** = PPO clip 裁剪示意 (第4章,可选)

## 已核实的硬事实 (只能用这些数字,不得杜撰)
- 实现代码 **2356 行** (agentic_rl/ 下全部 .py)
- 测试代码 **1489 行**；**115 个测试用例** = **113 passed + 2 skipped**;CPU **2.35s** 跑完
- 各测试文件用例数: test_00=8, test_01=10, test_02=11, test_03=20, test_04=17, test_05=14, test_06=10, test_07=7, test_08=5(3pass/2skip), test_09=9, test_10=4
- 6 个核心模块: env / rollout / credit / algo(+train) / shaping / train+configs
- 3 种信用分配打法: outcome(朴素) / R2G(UserRL 折扣前传) / InfoPO(反事实信息增益)
- 5 条真实 golden fixture: airline_success, airline_fail, retail_success, telecom_success_dualctrl, telecom_fail_dualctrl(取自 `Jarrodbarnes/tau2-sft-seed-v3`,带 gold reward_info)
- 冷启动模型 `Jarrodbarnes/Qwen3-4B-tau2-sft1`(pass@1 0.40,免 SFT 直接 RL 起点)；效果参照 `Jarrodbarnes/Qwen3-4B-tau2-grpo-v1`(pass@4 59%)
- 底座 tau2-bench: Sierra dual-control 客服 benchmark,retail/airline/telecom 三域;用户是活的 LLM,会 STOP/TRANSFER;工具有副作用;reward 靠数据库终态断言
- reward 语义: `reward = ∏ component[c] for c in reward_basis`(乘法门控),复刻 tau2 官方 evaluator;两种 basis: [DB,COMMUNICATE] 与 [ENV_ASSERTION]
- token-level loss masking: assistant token=1, obs token=0(逐 token 断言)
- R2G 公式: $R2G_t = r_t + \gamma R2G_{t+1}$,γ=0.8,对拍 survey TravelGym 算例第7轮=1.44
- InfoPO: $\text{gain}_t = \mathrm{KL}(P_\text{factual} \Vert P_\text{masked})$,零方差组(GRPO advantage≈0)靠 info-gain 救活供梯度
- BAO 正则①(Information-Seeking): 连续两轮都问用户 → 罚 $-\lambda_\text{ans}$
- BAO 正则②(Over-Thinking): 失败且提前终止 → 罚 $-\lambda_\text{think}\cdot(T-T')/T'$;对拍算例 4.33
- BAO λ_ans/λ_think/w 论文未公开数值,代码用占位默认(诚实边界)
- PPP 三目标: R = R_Prod + R_Proact + R_Pers;`[Cost N]`→effort 解析
- advantage estimators: GRPO(组内标准化 (r-μ)/σ) / RLOO(leave-one-out) / PPO+GAE
- DAPO 四件套: ①Clip-Higher(ε_high>ε_low) ②Dynamic Sampling(丢零方差组) ③Token-Level Loss ④Overlong Reward Shaping
- PPO loss: $\text{ratio}=\exp(\log\pi-\log\pi_\text{old})$, $L=-\min(\text{ratio}\cdot A,\ \text{clip}(\text{ratio},1-\epsilon,1+\epsilon_\text{high})\cdot A)$
- demo 训练实测: credit=r2g estimator=grpo gamma=0.8; success_logprob 从 step0 的 -6.3159 单调上升到 step29 的 -5.1066;mean_reward 恒 0.500;loss≈-0.0675
- 修过的真实 bug: R2G turn-level 归一化被错写在单条轨迹内 `traj_adv*(1+nv)`,把成功轨迹早期轮翻成负 advantage;正确做法是在整个 group 层面统一归一化;修完 logprob 从 -6.3 稳升到 -5.1
- 真机迁移(零算法改动): TinyTransformer→Qwen3-4B / ScriptedPolicy→HFPolicy / TauEnv→RealTau2Env / ByteTokenizer→Qwen3 tokenizer;换 YAML 即可
- 诚实边界: tau2 需 Python≥3.12(离线环境 3.9 装不上,故用 mock env + 真实 gold reward_info 交叉验证)
- 源码文件参考: env/reward.py, env/tau_env.py, rollout/rollout.py, credit/credit.py, credit/infopo_runtime.py, shaping/shaping.py, algo/advantage.py, algo/loss.py, algo/dapo.py, train/trainer.py, train/model.py, utils/types.py, utils/config.py

## SVG 图配色 (学术白底)
背景白/#f7f6f3;盒子描边 #333/#888;文字 #1a1a1a;强调用暗红 #8b2635、深绿 #1f6b3f、靛蓝 #2c5aa0;字体用 sans/mono。**不要**深色背景、不要霓虹渐变。线条细(1-1.5px),箭头简洁。像论文 Figure 而非 dashboard。

## 语气与质量
- 学术、克制、精确。每个论断尽量挂一个验证锚点(测试/对拍值/文件名)。
- 正文中文为主,术语英文。**正文不用 emoji**。
- 你写的是完整可直接嵌入的 HTML 片段,自包含、语法正确、标签闭合。
- 只输出该章 HTML,开头是该章 `<h2>`。第1章 agent 额外负责标题块+摘要(见其专属指令)。
