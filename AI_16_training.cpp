#include <iostream>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <fstream>
using namespace std;
#define DEP_MAX 4
//输入所需数据结构： 1. 边权图 
// 以输入层到隐含层为例 ： temp[i][j] 表示输入层第i个点到隐含层第j个点的权值
//一个网络包含3组边权值数组
#define LEN   8
#define SIZE_IN  64 //输入的大小
#define SIZE_1  16 //第一层的大小
#define SIZE_2  4  //第二层的大小
#define SIZE_OUT 1 // 输出层的大小
#define MOD 64		//边权大小限制
#define CANDIDATE 16 //参与的网络数
#define OFFSPRING 8 //每轮产生的后代数 = 每轮淘汰的父代数 选择的父代数
#define MUT 5
#define COM 85

int map[LEN][LEN] =
{
	0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,
	0,0,0,-1,1,0,0,0,
	0,0,0,1,-1,0,0,0,
	0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0
};
const int cost[8][8] =												//棋盘各点权值估值
{
	{ 90,-60,10,10,10,10,-60,90 },
	{ -60,-80,5,5,5,5,-80,-60 },
	{ 10,5,1,1,1,1,5,10 },
	{ 10,5,1,1,1,1,5,10 },
	{ 10,5,1,1,1,1,5,10 },
	{ 10,5,1,1,1,1,5,10 },
	{ -60,-80,5,5,5,5,-80,-60 },
	{ 90,-60,10,10,10,10,-60,90 }
};
int direction[8][2] =
{//前i 后j
	-1,-1,
	-1,0,
	-1,1,
	0,1,
	1,1,
	1,0,
	1,-1,
	0,-1
};
struct NN
{
	double Cost_In_1[SIZE_IN][SIZE_1],//三层权值
		Cost_1_2[SIZE_1][SIZE_2],
		Cost_2_Out[SIZE_2][SIZE_OUT];
	int In[LEN][LEN];//输入层 变换法则 x*8 + y 得到输入网络对应位置的权值
	double Hidden1[SIZE_1];
	double Hidden2[SIZE_2];
	double Out[SIZE_OUT];//输出层
	int Parent;//是否作为父母 1 是 0 不是
	int Win;//获胜的次数
	int Play[CANDIDATE];//标记对局情况 1 已经下过 0 还未下过
	int BlackOrWhite;//控制该网络的执子 黑色为1 白色-1 初始为0 (与棋盘中表示黑白的方式一致即可)
}ANN[CANDIDATE];
struct NN FinalNN;
int cmp(const void * n1, const void * n2)
{
	struct NN * n3 = (NN*)n1;
	struct NN * n4 = (NN*)n2;
	if ((*n3).Win > (*n4).Win) return -1;
	else return 1;
}
//得到随机概率 输入 - 模数 输出 - 小于模数的随机数
int GetRand(int mod)
{
	srand(time(NULL));
	int seeds;
	seeds = rand() % mod;
	//设模数为107 加数c = 13 乘数 a = 23
	for (int i = 0; i < 19; i++)
	{
		//seeds = (double)(23 * seeds + 13) % 107;
		int x;
		x = 23 * seeds + 13;
		//seeds = x % 107;
		x %= mod;
		seeds = x;
	}
	return seeds;
}
//判断是否落入 输入 - 是否落入的界限k, 模数 - mod 
//输出 1：落入 0：不落入
int Judge(int k, int mod)
{
	int tmp;
	tmp = GetRand(mod);
	if (tmp >= k) return 0;
	else return 1;
}
//神经网络部分
//初始化网络 输入 ： 需要初始化的网络 输出：初始化完毕的网络
struct NN Init(struct NN & nn, int order)//初始化一个网络的边权，大概需要多次调用
{
	//InitCost(nn.Cost_In_1,SIZE_IN,SIZE_1);
	nn.Parent = 0;
	nn.Win = 0;
	nn.BlackOrWhite = 0;
	for (int i = 0; i < CANDIDATE; i++)
	{
		nn.Play[i] = 0;
	}
	for (int i = 0; i < SIZE_1; i++)
	{
		nn.Hidden1[i] = 0;
		nn.Hidden2[i / SIZE_2] = 0;
		nn.Out[i / SIZE_1] = 0;
	}
	srand(time(NULL) + order);//随机化赋予边权值
	for (int i = 0; i < SIZE_IN; i++)
		for (int j = 0; j < SIZE_1; j++)
		{
			nn.Cost_In_1[i][j] = 2 * (1.0*(rand() % MOD - MOD / 2) / MOD);
		}
	//	srand(time(NULL));
	for (int i = 0; i < SIZE_1; i++)
		for (int j = 0; j < SIZE_2; j++)
		{
			nn.Cost_1_2[i][j] = 1.0 * 2 * (1.0*(rand() % MOD - MOD / 2) / MOD);
		}
	//	srand(time(NULL));
	for (int i = 0; i < SIZE_2; i++)
		for (int j = 0; j < SIZE_OUT; j++)
		{
			nn.Cost_2_Out[i][j] = 1.0 * 2 * (1.0*(rand() % MOD - MOD / 2) / MOD);
		}
	return nn;
}
//计算权值
//激活函数为 tanh  输入：需要计算的网络  输出：计算完毕的网络
//最终权值存储在输出层
struct NN InitFinalNN(struct NN & nn)//初始化一个网络的边权，大概需要多次调用
{
	//InitCost(nn.Cost_In_1,SIZE_IN,SIZE_1);
	nn.Parent = 0;
	nn.Win = 0;
	nn.BlackOrWhite = 0;
	for (int i = 0; i < CANDIDATE; i++)
	{
		nn.Play[i] = 0;
	}
	for (int i = 0; i < SIZE_1; i++)
	{
		nn.Hidden1[i] = 0;
		nn.Hidden2[i / SIZE_2] = 0;
		nn.Out[i / SIZE_1] = 0;
	}
	//(time(NULL) + order);//随机化赋予边权值
	fstream fcost("cost.txt");
	for (int i = 0; i < SIZE_IN; i++)
		for (int j = 0; j < SIZE_1; j++)
		{
			fcost >> nn.Cost_In_1[i][j];// = 2 * (1.0*(rand() % MOD - MOD / 2) / MOD);
		}
	//	srand(time(NULL));
	for (int i = 0; i < SIZE_1; i++)
		for (int j = 0; j < SIZE_2; j++)
		{
			fcost >> nn.Cost_1_2[i][j];// = 1.0 * 2 * (1.0*(rand() % MOD - MOD / 2) / MOD);
		}
	//	srand(time(NULL));
	for (int i = 0; i < SIZE_2; i++)
		for (int j = 0; j < SIZE_OUT; j++)
		{
			fcost >> nn.Cost_2_Out[i][j];// = 1.0 * 2 * (1.0*(rand() % MOD - MOD / 2) / MOD);
		}
	return nn;
}

struct NN Calculate(struct NN & nn)//计算权值
{
	for (int i = 0; i < SIZE_1; i++)
	{
		nn.Hidden1[i] = 0;
		for (int j = 0; j < SIZE_IN; j++)
		{
			nn.Hidden1[i] += nn.Cost_In_1[j][i] * nn.In[j / LEN][j%LEN]; //基于权值和输入矩阵求和
		}
		nn.Hidden1[i] = tanh(nn.Hidden1[i]);
		//printf("%lf ", nn.Hidden1[i]);
	}
	//printf("\n");
	for (int i = 0; i < SIZE_2; i++)
	{
		nn.Hidden2[i] = 0;
		for (int j = 0; j < SIZE_1; j++)
		{
			nn.Hidden2[i] += nn.Cost_1_2[j][i] * nn.Hidden1[j];
		}
		nn.Hidden2[i] = tanh(nn.Hidden2[i]);
		//	printf("%lf ", nn.Hidden2[i]);
	}
	//printf("\n");
	for (int i = 0; i < SIZE_OUT; i++)
	{
		nn.Out[i] = 0;
		for (int j = 0; j < SIZE_2; j++)
		{
			nn.Out[i] += nn.Cost_2_Out[j][i] * nn.Hidden2[j];
		}
		nn.Out[i] = tanh(nn.Out[i]);
		//printf("%lf ", nn.Out[i]);
	}
	return nn;
}
//输入网络需要计算的初始状态 输入：请求输入的网络， 状态图 输出：输入完毕的图
//从 map 输入初始状态
struct NN InputTrain(struct NN & nn, int map[][LEN])//向网络输入当前状态，
{
	for (int i = 0; i < LEN; i++)
		for (int j = 0; j < LEN; j++)
		{
			//scanf("%d", &nn.In[i][j]);
			nn.In[i][j] = map[i][j];
		}
	return nn;
}

double Evaluation(struct NN & nn, int map[][LEN], int turn)//nn应当已经具有权值 - 无论为初始化或选择完毕
{
	InputTrain(nn, map);//输入
	nn.BlackOrWhite = turn;
	Calculate(nn);
	double result;
	result = nn.Out[0];
	return result;
}


//基因算法部分 
// 适应度由竞争的胜率确定，不人为确定胜率
// 基因编码 SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT 为一个基因串
//突变函数 - 缺少突变函数
struct NN Mutation(struct NN & nn)
{
	for (int i = 0; i < SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT; i++)
	{
		if (!Judge(MUT, 107)) continue;//如果 判定不突变则不进行修改
		else
		{
			//根据突变修正函数进行处理
			int j;//处理每段内部的突变
			if (i < SIZE_IN*SIZE_1)//判定三段范围
			{
				j = i;
				nn.Cost_In_1[j / SIZE_1][j%SIZE_1] = 2 * (1.0*(GetRand(MOD) - MOD / 2) / MOD);//
			}
			else if (i < SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 && i >= SIZE_IN * SIZE_1)
			{
				j = i - SIZE_IN * SIZE_1;
				nn.Cost_1_2[j / SIZE_2][j%SIZE_2] = 2 * (1.0*(GetRand(MOD) - MOD / 2) / MOD);//
			}
			else
			{
				j = i - SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2;
				nn.Cost_2_Out[j / SIZE_OUT][j%SIZE_OUT] = 2 * (1.0*(GetRand(MOD) - MOD / 2) / MOD);//
			}
		}
	}
	return nn;
}
//组合函数 - 传参方式可能需要修改，替代父代的方式
//输入 2个父代 2个将要被替代的父代 结果：子代放入到被替代的父代中
void Combination(struct NN n1, struct NN n2, struct NN & n3, struct NN & n4)
{
	if (!Judge(COM, 107)) //判断是否重组
	{
		n3 = n1;
		n4 = n2;
	}
	else
	{
		//处理多余变量
		n4.Win = 0; n3.Win = 0;
		n4.Parent = 0; n3.Parent = 0;
		for (int i = 0; i < CANDIDATE; i++)
		{
			n3.Play[i] = 0;
			n4.Play[i] = 0;
		}
		int INTER1, INTER2;//判定两个交换断点
		INTER1 = (int)100 * GetRand(SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT);
		INTER2 = (int)100 * GetRand(SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT);
		if (INTER1 > INTER2)//两个断点保证 1 一定小于 2
		{
			int tmp;
			tmp = INTER1;
			INTER1 = INTER2;
			INTER2 = tmp;
		}
		for (int i = 0; i < SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT; i++)
		{
			//对 两个断点两端的部分进行交换重组操作
			int j; //进行内部定位
			if (i > INTER1 && i <= INTER2)//中间交换部分
			{
				if (i < SIZE_IN*SIZE_1)//判定三段范围
				{
					j = i;
					n3.Cost_In_1[j / SIZE_1][j%SIZE_1] = n2.Cost_In_1[j / SIZE_1][j%SIZE_1];
					n4.Cost_In_1[j / SIZE_1][j%SIZE_1] = n1.Cost_In_1[j / SIZE_1][j%SIZE_1];
				}
				else if (i < SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 && i >= SIZE_IN * SIZE_1)
				{
					j = i - SIZE_IN * SIZE_1;
					n3.Cost_1_2[j / SIZE_2][j%SIZE_2] = n2.Cost_1_2[j / SIZE_2][j % SIZE_2];
					n4.Cost_1_2[j / SIZE_2][j%SIZE_2] = n1.Cost_1_2[j / SIZE_2][j % SIZE_2];
				}
				else
				{
					j = i - SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2;
					n3.Cost_2_Out[j / SIZE_OUT][j%SIZE_OUT] = n2.Cost_2_Out[j / SIZE_OUT][j%SIZE_OUT];
					n4.Cost_2_Out[j / SIZE_OUT][j%SIZE_OUT] = n1.Cost_2_Out[j / SIZE_OUT][j%SIZE_OUT];
				}
			}
			else
			{
				if (i < SIZE_IN*SIZE_1)//判定三段范围
				{
					j = i;
					n3.Cost_In_1[j / SIZE_1][j%SIZE_1] = n1.Cost_In_1[j / SIZE_1][j%SIZE_1];
					n4.Cost_In_1[j / SIZE_1][j%SIZE_1] = n2.Cost_In_1[j / SIZE_1][j%SIZE_1];
				}
				else if (i < SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 && i >= SIZE_IN * SIZE_1)
				{
					j = i - SIZE_IN * SIZE_1;
					n3.Cost_1_2[j / SIZE_2][j%SIZE_2] = n1.Cost_1_2[j / SIZE_2][j % SIZE_2];
					n4.Cost_1_2[j / SIZE_2][j%SIZE_2] = n2.Cost_1_2[j / SIZE_2][j % SIZE_2];
				}
				else
				{
					j = i - SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2;
					n3.Cost_2_Out[j / SIZE_OUT][j%SIZE_OUT] = n1.Cost_2_Out[j / SIZE_OUT][j%SIZE_OUT];
					n4.Cost_2_Out[j / SIZE_OUT][j%SIZE_OUT] = n2.Cost_2_Out[j / SIZE_OUT][j%SIZE_OUT];
				}
			}
		}
	}
}
//直接以全局变量作为输入，结果为排序....并概率的选择 基于RR
void Selection()//依据概率选择数组中的值作为父代
{
	//排序确定先后顺序
	qsort(ANN, CANDIDATE, sizeof(ANN[1]), cmp);
	//求和得到 概率 
	int sum = 0;//和
	int threshold[16][2] = { 0 };//每一个的阈值
	for (int i = 0; i < CANDIDATE; i++)//计算阈值并求和
	{	//设置阈值均为开区间
		if (i == 0)
		{
			threshold[i][0] = 0;  threshold[i][1] = ANN[i].Win;
		}
		else
		{
			threshold[i][0] = threshold[i - 1][1];
			threshold[i][1] = threshold[i][0] + ANN[i].Win;
		}
		sum += ANN[i].Win;
	}
	//RR算法确定是否落入
	int count = 0;
	for (int i = 0; i < CANDIDATE; i++)
	{
		int gamble = 0;
		gamble = GetRand(sum + 1);
		if (gamble > threshold[i][0] && gamble < threshold[i][1])
		{
			ANN[i].Parent = 1;
			count++;
		}
		else
		{
			ANN[i].Parent = 0;
		}
		if (count >= OFFSPRING) break;
	}
}
//扫描棋盘判断谁输谁赢
int Check_Win(int map[][LEN])
{
	int num_black = 0, num_white = 0;
	for (int i = 0; i < LEN; i++)
		for (int j = 0; j < LEN; j++)
		{
			if (map[i][j] == 1) num_black++;
			else if (map[i][j] == -1) num_white++;
		}
	if (num_black > num_white) return 1;
	else return -1;
}
//用于搜索的数据结构,表示当前状态
//训练的初始过程
void Start(int first)
{
	for (int i = 0; i < CANDIDATE; i++) //初始所有的待选网络权值
	{
		if (first)		Init(ANN[i], i);//第一次时直接初始化
		else
		{
			ANN[i].Parent = 0;
			ANN[i].Win = 0;
			ANN[i].BlackOrWhite = 0;
			for (int i = 0; i < CANDIDATE; i++)
			{
				ANN[i].Play[i] = 0;
			}
			for (int i = 0; i < SIZE_1; i++)
			{
				ANN[i].Hidden1[i] = 0;
				ANN[i].Hidden2[i / SIZE_2] = 0;
				ANN[i].Out[i / SIZE_1] = 0;
			}
		}//其他情况不初始网络权值，只修正胜负次数等
	}
}
//下棋部分
//返回0则没有邻居 不可以在此落子 
int flagx[8], flagy[8], f_end = 0;//用于记录本次翻转位置 
struct pos_stack {
	int tmp_pos[60][2];
	int tmp_pend;
}Ps[8];
int p_top = 0;
int position[60][2], p_end = 0;//用记录可以走的点数 

int tmp_map[8][8][8];
int m_top = 0;

void Copy_Board(int tmp_map[][LEN], int map[LEN][LEN])
{
	for (int i = 0; i < LEN; i++)
		for (int j = 0; j < LEN; j++)
		{
			tmp_map[i][j] = map[i][j];
		}
}
void Copy_Position(int tmp_pos[60][2], int pos[][2], int &tmp_end, int pos_end)
{
	tmp_end = pos_end;
	for (int i = 0; i < pos_end; i++)
	{
		tmp_pos[i][0] = pos[i][0];
		tmp_pos[i][1] = pos[i][1];
	}
}
int Check_Neighbor(int x, int y, int map[LEN][LEN], int turn)
{
	int tmp_x, tmp_y, flag = 0;
	if (map[x][y] != 0) return 0;//不为空
	else if (x < 0 || x >= LEN || y < 0 || y >= LEN) return 0;//超出范围 
	f_end = 0;//清空翻转结束标记栈 
	for (int k = 0; k < 8; k++)
	{
		int mov_x = direction[k][0];
		int mov_y = direction[k][1];
		//至少有一个子可以翻 
		if (x + mov_x < 0 || x + mov_x >= LEN || y + mov_y < 0 || y + mov_y >= LEN) continue;
		if (map[x + mov_x][y + mov_y] == -1 * turn)
		{
			//找构成反转的另一子 
			tmp_x = x + 2 * mov_x;
			tmp_y = y + 2 * mov_y;
			while (1)//不为空时向后找，没有找到终点向后找 
			{
				//越界 
				if (tmp_x < 0 || tmp_x >= LEN || tmp_y < 0 || tmp_y >= LEN) break;
				else if (map[tmp_x][tmp_y] == 0)  break;
				else if (map[tmp_x][tmp_y] == turn)
				{
					flagx[f_end] = tmp_x;
					flagy[f_end] = tmp_y;
					f_end++;
					flag = 1;//找到 	
					break;
				}
				tmp_x += mov_x;
				tmp_y += mov_y;
			}
		}
	}
	return flag;
}
int CanMove(int map[LEN][LEN], int turn)
{
	int flag = 0;
	int final = 0;
	p_end = 0;//清零可移动位置 
	for (int i = 0; i < LEN; i++)
		for (int j = 0; j < LEN; j++)
		{
			flag = Check_Neighbor(i, j, map, turn);
			if (flag == 1)//可移动 
			{
				final = 1;
				position[p_end][0] = i;
				position[p_end][1] = j;
				p_end++;
			}
		}
	return final;
}
int Print_Win(int map[LEN][LEN])
{
	int result = 0;
	for (int i = 0; i < LEN; i++)
		for (int j = 0; j < LEN; j++)
		{
			result += map[i][j];
		}
	if (result > 0)return 1;
	else if (result < 0) return -1;
	else return 0;
}
//当无期可走 跳过 
int Skip(int & turn)
{

	if (turn == 1) 	printf("black skip\n");
	else printf("white skip\n");
	turn *= -1;
	return turn;
}
//下棋和翻转 
void Play(int x, int y, int turn)
{
	int flag = 0;

	while (!Check_Neighbor(x, y, map, turn))//当不和法时
	{
		printf("illegal\n");
		scanf("%d %d", &x, &y);
	}
	int tmp_x, tmp_y;
	int mov_x, mov_y;
	while (f_end)
	{
		f_end--;
		if (flagx[f_end] - x > 0) 			mov_x = 1;
		else if (flagx[f_end] - x == 0)		mov_x = 0;
		else if (flagx[f_end] - x < 0)		mov_x = -1;

		if (flagy[f_end] - y > 0) 			mov_y = 1;
		else if (flagy[f_end] - y == 0)		mov_y = 0;
		else if (flagy[f_end] - y < 0)		mov_y = -1;
		tmp_x = x;
		tmp_y = y;
		while (tmp_x != flagx[f_end] || tmp_y != flagy[f_end])//未到终点
		{
			if (tmp_x == x && tmp_y == y)  map[x][y] = turn;
			else map[tmp_x][tmp_y] *= -1;//翻转
			tmp_x += mov_x;
			tmp_y += mov_y;
		}
	}
}

void Output()
{
	printf(" ");
	for (int i = 0; i < LEN; i++)
	{
		printf(" %d", i);
	}
	printf("\n");

	for (int i = 0; i < LEN; i++)
	{
		printf("%d", i);
		for (int j = 0; j < LEN; j++)
		{
			if (map[i][j] < 0) printf("%d", map[i][j]);
			else 			printf(" %d", map[i][j]);
		}
		printf("\n");
	}
}
double Evaluation(int map[][LEN])
{
	double result = 0;
	for (int i = 0; i < LEN; i++)
		for (int j = 0; j < LEN; j++)
		{
			result += 1.0*cost[i][j] * map[i][j];
		}
	return result;
}

int best_x, best_y;
double Alpha_Beta(struct NN &nn,int map[][LEN], double alpha, double beta, int turn, int depth)
{
	double maxx = INT_MIN;
	if (depth <= 0)
		return turn * Evaluation(nn,map,turn);
	if (!CanMove(map, turn))
	{
		if (!CanMove(map, turn*-1))
			return turn * Evaluation(nn,map,turn);
		return -Alpha_Beta(nn, map,-1 * beta, -1 * alpha, -1 * turn, depth - 1);
	}
	//CanMove(map,turn)执行 position压栈完毕 
	while (p_end)
	{

		double value;
		//保护 
		Copy_Board(tmp_map[m_top], map);
		m_top++;
		Copy_Position(Ps[p_top].tmp_pos, position, Ps[p_top].tmp_pend, p_end);
		p_top++;
		//落子
		Play(position[p_end - 1][0], position[p_end - 1][1], turn);
		//递归 
		value = -Alpha_Beta(nn,map, -1 * beta, -1 * alpha, -1 * turn, depth - 1);
		//回溯 
		m_top -= 1;
		Copy_Board(map, tmp_map[m_top]);
		p_top -= 1;
		Copy_Position(position, Ps[p_top].tmp_pos, p_end, Ps[p_top].tmp_pend);
		//检查 
		if (value > alpha)
		{
			if (value >= beta)
				return value;
			alpha = value > alpha ? value : alpha;
		}
		if (value > maxx)
		{
			maxx = value;
			//记录最佳移动 
			if (depth == DEP_MAX)
			{
				best_x = position[p_end - 1][0];
				best_y = position[p_end - 1][1];
			}
		}
		p_end--;
	}
	return maxx;
}
int PcPlay(int & turn,struct NN &nn)
{
	double alpha = INT_MIN;
	double beta = INT_MAX;
	Alpha_Beta(nn,map, alpha, beta, turn, DEP_MAX);
	printf("%d %d\n", best_x, best_y);
	Check_Neighbor(best_x, best_y, map, turn);//找到标记 
	Play(best_x, best_y, turn);
	turn *= -1;
	return turn;
}
int ManPlay(int &turn)
{
	int x, y;
	scanf("%d %d", &x, &y);
	Play(x, y, turn);
	turn *= -1;
	return turn;
}
int main()
{
	int turn = 1;
	int no_more = 0;
	int player, PC;
	int Mode = 0;
	printf("Training Mode : 1\nP2P:2\nP2C : 3\n");
	scanf("%d", &Mode);
	if (Mode == 3)
	{
		printf("black : 1\nwhite : -1\n");
		scanf("%d", &player);
		PC = player*-1;
	}
	if (Mode != 1)
	{
		while (1)//训练模式不使用正常过程 
		{

			Output();
			int flag;
			flag = CanMove(map, turn);//检查是否有可行棋步 
			if (flag == 0)
			{
				no_more++;
				if (no_more == 2) break; //双方都无棋可下 
				turn = Skip(turn);
				continue;
			}
			no_more = 0;
			if (Mode == 3)
			{
				if (turn == 1)
				{
					if (turn == player)			ManPlay(turn);
					else 						PcPlay(turn,FinalNN);
				}
				else
				{
					if (turn == player)			ManPlay(turn);
					else 						PcPlay(turn,FinalNN);
				}
			}
			else if (Mode == 2)
			{
				ManPlay(turn);
			}
			else if (Mode == 1)
			{
				PcPlay(turn,FinalNN);
			}
		}
		int result;
		result = Print_Win(map);
		if (result == 1) printf("black win\n");
		else if (result == -1) printf("white win\n");
		else printf("fair\n");
	}
	else//训练模式 
	{
		//int t;
		Start(1);
		printf("Input Training times\n");
		//scanf("%d", &t);//本次训练的数据量
		ifstream fin("input.txt");
		ofstream fout("output.txt");
		int count = 0;
		while (!fin.eof())
		{
			count++;
			Start(0);
			if (fin)
			{//输入本次训练集
				fin >> turn;
				for (int i = 0; i < LEN; i++)
					for (int j = 0; j < LEN; j++)
					{
						fin >> map[i][j];
					}
				int tmp[LEN][LEN];
				for(int i = 0 ; i < LEN ; i ++ )
					for (int j = 0; j < LEN; j++)
					{
						tmp[i][j] = map[i][j];
					}
				//i 与 j对弈 黑白子随机
				for(int i = 0 ; i < CANDIDATE ;i ++ )
					for(int j = i+1 ; j < CANDIDATE ; j ++ )
					{
						int Coin = GetRand(100);
						//大于时 i黑字否则j黑子
						if (Coin > 50)
						{
							ANN[i].BlackOrWhite = 1;
							ANN[j].BlackOrWhite = -1;
						}
						else
						{
							ANN[i].BlackOrWhite = -1;
							ANN[j].BlackOrWhite = 1;
						}
						while (1)
						{
							Output();
							int flag;
							flag = CanMove(map, turn);//检查是否有可行棋步 
							if (flag == 0)
							{
								no_more++;
								if (no_more == 2) break; //双方都无棋可下 
								turn = Skip(turn);
								continue;
							}
							no_more = 0;
							if(turn == ANN[i].BlackOrWhite)			PcPlay(turn, ANN[i]);
							else									PcPlay(turn, ANN[j]);
						}
						if (Check_Win(map)) ANN[i].Win += 1;
						else ANN[j].Win += 1;
						for (int i = 0; i < LEN; i++)
							for (int j = 0; j < LEN; j++)
							{
								map[i][j] = tmp[i][j];
							}
					}
				//进行选择交换
				Selection();
				for (int i = 0; i < OFFSPRING - 2; i += 2)
				{
					Combination(ANN[i], ANN[i + 1], ANN[CANDIDATE - i - 2], ANN[CANDIDATE - i - 1]);
					Mutation(ANN[CANDIDATE - i - 2]);
					Mutation(ANN[CANDIDATE - i - 1]);
				}
			}
		}
		//写入将final写入文件
		Selection();
		FinalNN = ANN[0];
		for (int i = 0; i < SIZE_IN; i++)
		{
			for (int j = 0; j < SIZE_1; j++)
			{
				if (FinalNN.Cost_In_1[i][j] >= 0)	fout << " " << FinalNN.Cost_In_1[i][j];
				else 				fout << FinalNN.Cost_In_1[i][j];
			}
			fout << endl;
		}
		for (int i = 0; i < SIZE_1; i++)
		{
			for (int j = 0; j < SIZE_2; j++)
			{
				if (FinalNN.Cost_1_2[i][j] >= 0)		fout << " " << FinalNN.Cost_1_2[i][j];
				else                                    fout << FinalNN.Cost_1_2[i][j];
			}
			fout << endl;
		}
		for (int i = 0; i < SIZE_2; i++)
		{
			for (int j = 0; j < SIZE_OUT; j++)
			{
				if (FinalNN.Cost_2_Out[i][j] >= 0)	fout << " " << FinalNN.Cost_2_Out[i][j];
				else								fout << FinalNN.Cost_2_Out[i][j];
			}
			fout << endl;
		}
		fin.close();
		fout.close();
		printf("trainning data : %d\n",count);
	}
	std::system("pause");
	return 0;
}