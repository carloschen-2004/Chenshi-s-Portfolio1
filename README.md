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
<img width = "300" height = "350" alt = "image" src = "https://github.com/user-attachments/assets/fccf5671-f4a3-4df6-9190-a07386dd1963" />

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



## 核心技能与兴趣爱好 Core Skills and Interests
- **TA经历**：2025.02-2025.06，担任管工系廖颖老师(特聘研究员)的 R语言数据分析 课程助教，具有与导师良好的沟通、办事能力
- **优秀课设**:《Analysis and visualization of NBA data》:对NBA数据进行PCA/t-SNE/Umap降维;DBSCAN聚类GBDT/SVM；《构建明日社会：房价、就业与生育率的协同演变路径》:VAR/VECM模型、脉冲响应和方差分解。
- **智能优化算法设计**：采用TS+SA+多种邻域操作解决复杂的CBRPTW问题，得到了专业前四的排名。
- **技能**：英语CET-6(612),普通话(一级乙等) ; Python(Gurobi\ML\DL),R(熟练),SQL(熟练),SPSS ; Linux ; Office ; Tableau ; C
- **兴趣爱好**：健身,篮球,龙舟(武汉大学龙舟队 副队长: 2023年 昆明全国大学生龙舟锦标赛团体三等奖等等),阅读,音乐,AI,Web3
  

- **TA experience**: 2025.02-2025.06, served as an assistant teacher for **R language data analysis course** of teacher Liao Ying (distinguished researcher) of the Department of Management Engineering, with good communication and work ability with the tutor
-  **Excellent course design**: "Analysis and visualization of NBA data": **PCA/t-SNE/Umap** dimension reduction of NBA data; **DBSCAN** clustering **GBDT/SVM**.
"Building Tomorrow's Society: Co-evolution Path of Housing Prices, Employment and Fertility": **VAR/VECM** model, impulse response and variance decomposition.
- **Intelligent optimization algorithm design**: **TS+SA+multiple neighborhood operations** are used to solve complex **CVRPTW** problems, and the professional ranking is ranked in the **top four in the class**
- **Skills**: English CET-6 (612), Mandarin (Level 1 B); **Python** (Gurobi\ML\DL), R (proficient),**SQL** (proficient), SPSS; Linux; Office; Tableau; C; C++
- **Hobbies**: Fitness, basketball, dragon boat (**Wuhan University Dragon Boat Team Vice Captain**: 2023 Kunming National College Student Dragon Boat Championship Team Third Prize, etc.), reading, music, AI, Web3

