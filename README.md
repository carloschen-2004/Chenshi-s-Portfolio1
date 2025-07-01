# Chenshi-s-Portfolio
武汉大学 管理科学与工程 陈实 的作品集，涵盖（1）**商科**（2）**数据科学与数据分析** 领域的一些微薄成绩，包括竞赛、课设作品等等。

Wuhan University｜Management Science and Engineering｜Chen Shi's portfolio covers (1) **Business** and (2) **Data Science and Data Analysis**, including some modest achievements in competitions, course projects, etc.

**联系方式[CONTACTS]**

- **Email**: 431684123@qq.com
- **Tele｜Wechat:** 18851726279


## 曾获荣誉 Awards
全国大学生数学竞赛**省级一等奖**、美赛**F奖(1%)**、华中杯**省级二等奖**、数维杯**H奖**;商挑精英品牌策划大赛**国家二等奖**、国际大学生创新大赛**校三等奖**、清华经管“今经乐道”商业分析大赛**全国二十五强**; 校**丙等**奖学金、国家**励志**奖学金、**优秀**学生。

- **1st prize at the provincial level** in the 2024 National College Mathematics Competition
-  **Finalist prize（Top 1%）** in the 2025 American Mathematics Competition
-  **2nd prize at the provincial level** in the 2025 Central China Cup Mathematics Competition
-  **Honerable prize** in the 2024 Shuwei Cup Mathematics Competition
-  **2nd prize at the national level** in the Business Selection Elite Brand Planning Competition
-  **3rd prize at the school level** in the International College Student Innovation Competition
-  **1st of the top 25 in the national** Tsinghua School of Economics and Management "Jinjing Ledao" Business Analysis Competition
-  **3rd-class** school scholarship, **national inspirational** scholarship, and **outstanding student** scholarship

## 数据科学&数据分析 Data Science & Data Analysis
### [**2025年美国大学生数学建模大赛 Finalist(1%) 01/2025**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%B1%BB/2025%E5%B9%B4%E7%BE%8E%E5%9B%BD%E5%A4%A7%E5%AD%A6%E7%94%9F%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%E6%AF%94%E8%B5%9B)  
<img width="300" alt="image" src="https://github.com/user-attachments/assets/93c83c6f-81c1-4b06-b429-6484f9e16b48"  />

[**比赛描述**] C 题：预测 2028 年洛杉矶奥运会奖牌数 
- 数据挖掘：识别奖牌分布过度离散，零点数据多（zero-inflation）；国家之间样本量差距较大。
- 特征工程：对齐历史与现在的数据(存在改名/消失问题)、构建国家参与次数、运动员参赛次数、过去三届获奖率、国家相似度等.
- 构建模型：构建“zero-inflated-负二项回归”的基准模型和“zero-inflated-XGBOOST”捕捉大国非线性特征、“zero-inflated-逻辑回归”预测小样本的小国奖牌数。 
- 参数优化：通过 Bayesian Optimation 对 XGBOOST 超参数优化，解决 XGBOOST 参数过多的问题。


### [**2025年华中杯数学建模大赛 省级二等奖 05/2025**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%B1%BB/2025%E5%B9%B4%E5%8D%8E%E4%B8%AD%E6%9D%AF%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%E6%AF%94%E8%B5%9B)
<img width="400" alt="image" src="https://github.com/user-attachments/assets/418c3de1-6a16-4571-9047-e979ff5c5ebf" />

[**比赛概述**] B 题：校园共享单车的调度与维护问题
- 总量与分布估计：采用均值填补与同类型点位趋势估算方法处理缺失值;结合最小二乘法和 K-means 聚类估算总量并优化流动性；利用三次样条插值法生成高粒度时间序列分布。
- 需求预测与调度:结合 Dijkstra 算法与 OpenCV 人机交互地进行坐标提取，采用 CVRPTC-SA 优化算法最小化调度总时间。
- 参数优化：构建时间-空间网络流模型并运用 Gurobi 线性规划优化调度,增加供需不平衡惩罚,综合评估调度效率。


### [**2024年数维杯国际数学建模大赛 Honorable 11/2024**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%B1%BB/2024%E5%B9%B4%E7%AC%AC%E5%8D%81%E5%B1%8A%E6%95%B0%E7%BB%B4%E6%9D%AF%E5%9B%BD%E9%99%85%E5%A4%A7%E5%AD%A6%E7%94%9F%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%E6%8C%91%E6%88%98%E8%B5%9B)
<img width="400" height = "250" alt="image" src="https://github.com/user-attachments/assets/1c778aa6-3692-4c8d-9c06-104ad80390eb" />

[**比赛概述**] B 题：空间变量协同估计方法研究
- 数据特征识别：自相关性;运用 Moran's I 指数和 Geary's C 指数进行空间自相关检验，验证采样方法有效性；协相关性：构建Bivariate Moran's I 空间相关性评估框架，筛选高协同性协变量。
- 解决数据稀疏性：提出“克里金预填充 + 深度插值”策略，解决 F2 变量稀疏问题，插值精度提升 10%。
- 创新模型：开发神经网络克里金算法，在高采样率下误差较协同克里金降低 8%。


### [**招商银行数据赛道夏令营 个人参赛 05/2025**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%B1%BB/%E6%8B%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E8%B5%9B%E9%81%93%E5%A4%8F%E4%BB%A4%E8%90%A5)
<img width="250" alt="image" src="https://github.com/user-attachments/assets/447e189a-1bbc-40e4-8128-00a7d35f5ecc" />

- 广告 CTR 预测模型:神深特征工程;使用 LightGBM 进行训练和交叉验证，最终预测 AUC 为 0.609(与第一名仅差 0.003)
- 文本分类：通过 TF-IDF 和 SVD 降维进行特征工程，并结合 XGBoost 模型结合分层交叉验证进行训练,F1-Score:0.809(前 20%)
- 营销智能体：调用 LLM 接口(qwen-2.5)设计智能营销 Agent，根据客户背景推荐合适理财产品并生成个性化营销话术。


### [**数据探索与可视化 课程设计 12/2024**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%B1%BB/%E6%95%B0%E6%8D%AE%E6%8E%A2%E7%B4%A2%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96)
<img width="350" alt="image" src="https://github.com/user-attachments/assets/d058e062-8e9e-45b4-b683-dac73214f507" />

《**Analysis and visualization of NBA data**》

使用NBA数据集进行数据探索，PCA、t-SNE、Umap降维，Kmeans、DBSCAN聚类，神经网络，梯度提升决策数，SVM进行可视化分析 ，展示数据的内在结构、聚类分布以及分类效果，从而深入洞察 NBA 球员、球队数据背后的规律与特征。


### [**时间序列分析 课程设计 12/2024**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%B1%BB/%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E5%88%86%E6%9E%90)
<img width="400" alt="image" src="https://github.com/user-attachments/assets/79fe087f-0c54-46dc-a82a-97578e93a676" />

《**构建明日社会：房价、就业与生育率的协同演变路径**》

通过构建向量自回归模型（VAR）、向量误差修正模型（VECM）， 结合脉冲响应分析与方差分解，分析了房价、失业率以及受教育水平对出生率的长期动态交互效应,为政策制定与学术研究提供支撑。

### [**智能优化算法 课程设计 05/2025**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%B1%BB/%E6%99%BA%E8%83%BD%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95)
<img width="340" alt="image" src="https://github.com/user-attachments/assets/cbec718b-9d89-4c64-bf66-63f8c5a20e0a" />

针对复杂带时间窗的车辆路径规划问题（CVRPTW，Capacitated Vehicle Routing Problem with Time Windows ），我创新性构建混合算法框架，融合禁忌搜索（TS，Tabu Search ）、模拟退火（SA，Simulated Annealing ）算法，并搭配多种邻域操作策略，高效攻克该 NP 难问题，排名专业第四。

### [**机器学习与预测 课程设计 05/2025**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%B1%BB/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%8E%E9%A2%84%E6%B5%8B)
<img width="600" alt="image" src="https://github.com/user-attachments/assets/dba6c693-e2b0-4f0f-a93c-5d73305a5ef2" />

《**基于数据科学家薪水情况的机器学习与预测**》

基于Kaggle平台2023年国外数据科学家薪水数据集，运用线性回归、神经网络、支持向量机及随机森林等多种机器学习模型，对数据科学家薪资影响因素进行分析与预测，通过特征工程、超参数调优及模型性能对比，揭示经验水平、工作类型等特征对薪资的影响规律。

### [**Kaggle 经典竞赛 2024-2025**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%B1%BB/kaggle)
<img width="800" alt="image" src="https://github.com/user-attachments/assets/bfa92bc3-e7f6-41ef-89c6-c894a71adf29" />

包含：（1）泰坦尼克号幸存者预测竞赛 RIP （2）罗斯曼商店销售预测竞赛


## 商业科学 & 商业分析  Business Science & Business Analysis
### [**商挑精英——品牌策划大赛 国家级二等奖 2025**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E5%95%86%E7%A7%91%E7%B1%BB/%E5%95%86%E6%8C%91%E7%B2%BE%E8%8B%B1%E2%80%94%E2%80%94%E5%93%81%E7%89%8C%E7%AD%96%E5%88%92%E5%A4%A7%E8%B5%9B)
<img width="600" alt="image" src="https://github.com/user-attachments/assets/f2e75a62-a9f1-4e7d-a884-22e0bc021e30" />

《**正兴染坊 蓝印花布品牌策划书**》

[**比赛概述**] 对南通市非遗工艺“蓝印花布”正兴染坊进行实践调研;分析品牌现状、环境优势与挑战，提出了全面的品牌发展战略。
- 品牌战略与定位：通过 SWOT｜PEST｜五力模型 评估“正兴染坊”在工艺、渠道及非遗地位上的优劣势，识别市场机遇与挑战。
- 营销与传播：构建 AI 驱动的蓝印花布定制网页;采用 KOL+KOC 组合营销，构建全媒体营销矩阵，涵盖抖音/小红书等平台。
- 运营规划:采用 AARRR 模型明确财务预算、保本分析与预期收益，确保项目可行性。

### [**第三十届康腾案例分析大赛 决赛入围 2024**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E5%95%86%E7%A7%91%E7%B1%BB/%E7%AC%AC%E4%B8%89%E5%8D%81%E5%B1%8A%E5%BA%B7%E8%85%BE%E6%A1%88%E4%BE%8B%E5%88%86%E6%9E%90%E5%A4%A7%E8%B5%9B)
<img width="500" alt="image" src="https://github.com/user-attachments/assets/e578b22c-863e-4179-a36e-0568b75b0891" />

《**智能新纪元——如何利用AI推进工业革命**》

[**比赛概述**] 聚焦AI技术驱动第四次工业革命的商业落地，通过产业链分析、政策解构及案例建模，探索 AI 技术在产业升级中的商业化路径，为企业战略布局提供数据支撑与决策参考。
- 分层分析：构建“基础层-技术层-应用层”商业模型，发现应用层利润率（17.34%）显著高于基础层与技术层，剖析基础层核心资源竞争格局，为供应链策略提供依据。    
- 环境评估：运用PEST模型和DID双差分模型，量化分析政策、经济等因素对企业AI渗透度及技术投入的驱动效应。    
- 战略解构：以百度智能驾驶为例，通过波士顿矩阵分析业务商业价值，拆解全栈布局的研发与商业化平衡策略，提炼技术商业化路径。    

### [**清华经管主办“今经乐道”案例分析大赛 全国二十五强 2023**](https://github.com/carloschen-2004/Chenshi-s-Portfolio1/tree/main/%E5%95%86%E7%A7%91%E7%B1%BB/%E6%B8%85%E5%8D%8E%E7%BB%8F%E7%AE%A1%E4%B8%BB%E5%8A%9E%E2%80%9C%E4%BB%8A%E7%BB%8F%E4%B9%90%E9%81%93%E2%80%9D%E6%A1%88%E4%BE%8B%E5%88%86%E6%9E%90%E5%A4%A7%E8%B5%9B)
<img width="500" alt="image" src="https://github.com/user-attachments/assets/19aa998c-03fc-4ef5-9c53-0e6a59f9ffd5" />

《**以人为本——AI赋能智能家居**》

[**比赛概述**] 聚焦智能家居行业技术演进与商业化落地，通过产业链分析、案例解构及战略规划，探索 AI 技术在智能家居场景中的用户需求适配与产业升级路径。
- 分层分析：构建 “上游技术层（芯片 / 传感器）- 中游应用层（设计制造）- 下游消费市场（To B/To C）” 的产业链模型。
- 环境评估：从 PEST 角度分析了行业外部环境，指出发展阻力，如数据安全、技术、成本、行业标准、社会合作等方面的问题。
- 案例分析：以小米公司为例进行案例分析，介绍其主营业务、发展阶段、SWOT 分析、核心战略、供应链、竞争环境及特点等。


## 核心技能与兴趣爱好 Core Skills and Interests
- **TA经历**：2025.02-2025.06，担任管工系廖颖老师(特聘研究员)的 R语言数据分析 课程助教，具有与导师良好的沟通、办事能力
- **技能**：英语CET-6(612),普通话(一级乙等) ; Python(Gurobi\ML\DL),R(熟练),SQL(熟练),SPSS ; Linux ; Office ; Tableau ; C ; C++
- **兴趣爱好**：健身,篮球,龙舟(**武汉大学龙舟队 副队长**: 2023年 昆明全国大学生龙舟锦标赛团体三等奖等等),阅读,音乐,AI,Web3
  

- **TA experience**: 2025.02-2025.06, served as an assistant teacher for **R language data analysis course** of teacher Liao Ying (distinguished researcher) of the Department of Management Engineering, with good communication and work ability with the tutor
-  **Excellent course design**: "Analysis and visualization of NBA data": **PCA/t-SNE/Umap** dimension reduction of NBA data; **DBSCAN** clustering **GBDT/SVM**.
"Building Tomorrow's Society: Co-evolution Path of Housing Prices, Employment and Fertility": **VAR/VECM** model, impulse response and variance decomposition.
- **Intelligent optimization algorithm design**: **TS+SA+multiple neighborhood operations** are used to solve complex **CVRPTW** problems, and the professional ranking is ranked in the **top four in the class**
- **Skills**: English CET-6 (612), Mandarin (Level 1 B); **Python** (Gurobi\ML\DL), R (proficient),**SQL** (proficient), SPSS; Linux; Office; Tableau; C; C++
- **Hobbies**: Fitness, basketball, dragon boat (**Wuhan University Dragon Boat Team Vice Captain**: 2023 Kunming National College Student Dragon Boat Championship Team Third Prize, etc.), reading, music, AI, Web3

