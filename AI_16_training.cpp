#include <iostream>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <fstream>
using namespace std;
#define DEP_MAX 4
//�����������ݽṹ�� 1. ��Ȩͼ 
// ������㵽������Ϊ�� �� temp[i][j] ��ʾ������i���㵽�������j�����Ȩֵ
//һ���������3���Ȩֵ����
#define LEN   8
#define SIZE_IN  64 //����Ĵ�С
#define SIZE_1  16 //��һ��Ĵ�С
#define SIZE_2  4  //�ڶ���Ĵ�С
#define SIZE_OUT 1 // �����Ĵ�С
#define MOD 64		//��Ȩ��С����
#define CANDIDATE 16 //�����������
#define OFFSPRING 8 //ÿ�ֲ����ĺ���� = ÿ����̭�ĸ����� ѡ��ĸ�����
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
const int cost[8][8] =												//���̸���Ȩֵ��ֵ
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
{//ǰi ��j
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
	double Cost_In_1[SIZE_IN][SIZE_1],//����Ȩֵ
		Cost_1_2[SIZE_1][SIZE_2],
		Cost_2_Out[SIZE_2][SIZE_OUT];
	int In[LEN][LEN];//����� �任���� x*8 + y �õ����������Ӧλ�õ�Ȩֵ
	double Hidden1[SIZE_1];
	double Hidden2[SIZE_2];
	double Out[SIZE_OUT];//�����
	int Parent;//�Ƿ���Ϊ��ĸ 1 �� 0 ����
	int Win;//��ʤ�Ĵ���
	int Play[CANDIDATE];//��ǶԾ���� 1 �Ѿ��¹� 0 ��δ�¹�
	int BlackOrWhite;//���Ƹ������ִ�� ��ɫΪ1 ��ɫ-1 ��ʼΪ0 (�������б�ʾ�ڰ׵ķ�ʽһ�¼���)
}ANN[CANDIDATE];
struct NN FinalNN;
int cmp(const void * n1, const void * n2)
{
	struct NN * n3 = (NN*)n1;
	struct NN * n4 = (NN*)n2;
	if ((*n3).Win > (*n4).Win) return -1;
	else return 1;
}
//�õ�������� ���� - ģ�� ��� - С��ģ���������
int GetRand(int mod)
{
	srand(time(NULL));
	int seeds;
	seeds = rand() % mod;
	//��ģ��Ϊ107 ����c = 13 ���� a = 23
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
//�ж��Ƿ����� ���� - �Ƿ�����Ľ���k, ģ�� - mod 
//��� 1������ 0��������
int Judge(int k, int mod)
{
	int tmp;
	tmp = GetRand(mod);
	if (tmp >= k) return 0;
	else return 1;
}
//�����粿��
//��ʼ������ ���� �� ��Ҫ��ʼ�������� �������ʼ����ϵ�����
struct NN Init(struct NN & nn, int order)//��ʼ��һ������ı�Ȩ�������Ҫ��ε���
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
	srand(time(NULL) + order);//����������Ȩֵ
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
//����Ȩֵ
//�����Ϊ tanh  ���룺��Ҫ���������  �����������ϵ�����
//����Ȩֵ�洢�������
struct NN InitFinalNN(struct NN & nn)//��ʼ��һ������ı�Ȩ�������Ҫ��ε���
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
	//(time(NULL) + order);//����������Ȩֵ
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

struct NN Calculate(struct NN & nn)//����Ȩֵ
{
	for (int i = 0; i < SIZE_1; i++)
	{
		nn.Hidden1[i] = 0;
		for (int j = 0; j < SIZE_IN; j++)
		{
			nn.Hidden1[i] += nn.Cost_In_1[j][i] * nn.In[j / LEN][j%LEN]; //����Ȩֵ������������
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
//����������Ҫ����ĳ�ʼ״̬ ���룺������������磬 ״̬ͼ �����������ϵ�ͼ
//�� map �����ʼ״̬
struct NN InputTrain(struct NN & nn, int map[][LEN])//���������뵱ǰ״̬��
{
	for (int i = 0; i < LEN; i++)
		for (int j = 0; j < LEN; j++)
		{
			//scanf("%d", &nn.In[i][j]);
			nn.In[i][j] = map[i][j];
		}
	return nn;
}

double Evaluation(struct NN & nn, int map[][LEN], int turn)//nnӦ���Ѿ�����Ȩֵ - ����Ϊ��ʼ����ѡ�����
{
	InputTrain(nn, map);//����
	nn.BlackOrWhite = turn;
	Calculate(nn);
	double result;
	result = nn.Out[0];
	return result;
}


//�����㷨���� 
// ��Ӧ���ɾ�����ʤ��ȷ��������Ϊȷ��ʤ��
// ������� SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT Ϊһ������
//ͻ�亯�� - ȱ��ͻ�亯��
struct NN Mutation(struct NN & nn)
{
	for (int i = 0; i < SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT; i++)
	{
		if (!Judge(MUT, 107)) continue;//��� �ж���ͻ���򲻽����޸�
		else
		{
			//����ͻ�������������д���
			int j;//����ÿ���ڲ���ͻ��
			if (i < SIZE_IN*SIZE_1)//�ж����η�Χ
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
//��Ϻ��� - ���η�ʽ������Ҫ�޸ģ���������ķ�ʽ
//���� 2������ 2����Ҫ������ĸ��� ������Ӵ����뵽������ĸ�����
void Combination(struct NN n1, struct NN n2, struct NN & n3, struct NN & n4)
{
	if (!Judge(COM, 107)) //�ж��Ƿ�����
	{
		n3 = n1;
		n4 = n2;
	}
	else
	{
		//����������
		n4.Win = 0; n3.Win = 0;
		n4.Parent = 0; n3.Parent = 0;
		for (int i = 0; i < CANDIDATE; i++)
		{
			n3.Play[i] = 0;
			n4.Play[i] = 0;
		}
		int INTER1, INTER2;//�ж����������ϵ�
		INTER1 = (int)100 * GetRand(SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT);
		INTER2 = (int)100 * GetRand(SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT);
		if (INTER1 > INTER2)//�����ϵ㱣֤ 1 һ��С�� 2
		{
			int tmp;
			tmp = INTER1;
			INTER1 = INTER2;
			INTER2 = tmp;
		}
		for (int i = 0; i < SIZE_IN * SIZE_1 + SIZE_1 * SIZE_2 + SIZE_2 * SIZE_OUT; i++)
		{
			//�� �����ϵ����˵Ĳ��ֽ��н����������
			int j; //�����ڲ���λ
			if (i > INTER1 && i <= INTER2)//�м佻������
			{
				if (i < SIZE_IN*SIZE_1)//�ж����η�Χ
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
				if (i < SIZE_IN*SIZE_1)//�ж����η�Χ
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
//ֱ����ȫ�ֱ�����Ϊ���룬���Ϊ����....�����ʵ�ѡ�� ����RR
void Selection()//���ݸ���ѡ�������е�ֵ��Ϊ����
{
	//����ȷ���Ⱥ�˳��
	qsort(ANN, CANDIDATE, sizeof(ANN[1]), cmp);
	//��͵õ� ���� 
	int sum = 0;//��
	int threshold[16][2] = { 0 };//ÿһ������ֵ
	for (int i = 0; i < CANDIDATE; i++)//������ֵ�����
	{	//������ֵ��Ϊ������
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
	//RR�㷨ȷ���Ƿ�����
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
//ɨ�������ж�˭��˭Ӯ
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
//�������������ݽṹ,��ʾ��ǰ״̬
//ѵ���ĳ�ʼ����
void Start(int first)
{
	for (int i = 0; i < CANDIDATE; i++) //��ʼ���еĴ�ѡ����Ȩֵ
	{
		if (first)		Init(ANN[i], i);//��һ��ʱֱ�ӳ�ʼ��
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
		}//�����������ʼ����Ȩֵ��ֻ����ʤ��������
	}
}
//���岿��
//����0��û���ھ� �������ڴ����� 
int flagx[8], flagy[8], f_end = 0;//���ڼ�¼���η�תλ�� 
struct pos_stack {
	int tmp_pos[60][2];
	int tmp_pend;
}Ps[8];
int p_top = 0;
int position[60][2], p_end = 0;//�ü�¼�����ߵĵ��� 

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
	if (map[x][y] != 0) return 0;//��Ϊ��
	else if (x < 0 || x >= LEN || y < 0 || y >= LEN) return 0;//������Χ 
	f_end = 0;//��շ�ת�������ջ 
	for (int k = 0; k < 8; k++)
	{
		int mov_x = direction[k][0];
		int mov_y = direction[k][1];
		//������һ���ӿ��Է� 
		if (x + mov_x < 0 || x + mov_x >= LEN || y + mov_y < 0 || y + mov_y >= LEN) continue;
		if (map[x + mov_x][y + mov_y] == -1 * turn)
		{
			//�ҹ��ɷ�ת����һ�� 
			tmp_x = x + 2 * mov_x;
			tmp_y = y + 2 * mov_y;
			while (1)//��Ϊ��ʱ����ң�û���ҵ��յ������ 
			{
				//Խ�� 
				if (tmp_x < 0 || tmp_x >= LEN || tmp_y < 0 || tmp_y >= LEN) break;
				else if (map[tmp_x][tmp_y] == 0)  break;
				else if (map[tmp_x][tmp_y] == turn)
				{
					flagx[f_end] = tmp_x;
					flagy[f_end] = tmp_y;
					f_end++;
					flag = 1;//�ҵ� 	
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
	p_end = 0;//������ƶ�λ�� 
	for (int i = 0; i < LEN; i++)
		for (int j = 0; j < LEN; j++)
		{
			flag = Check_Neighbor(i, j, map, turn);
			if (flag == 1)//���ƶ� 
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
//�����ڿ��� ���� 
int Skip(int & turn)
{

	if (turn == 1) 	printf("black skip\n");
	else printf("white skip\n");
	turn *= -1;
	return turn;
}
//����ͷ�ת 
void Play(int x, int y, int turn)
{
	int flag = 0;

	while (!Check_Neighbor(x, y, map, turn))//�����ͷ�ʱ
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
		while (tmp_x != flagx[f_end] || tmp_y != flagy[f_end])//δ���յ�
		{
			if (tmp_x == x && tmp_y == y)  map[x][y] = turn;
			else map[tmp_x][tmp_y] *= -1;//��ת
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
	//CanMove(map,turn)ִ�� positionѹջ��� 
	while (p_end)
	{

		double value;
		//���� 
		Copy_Board(tmp_map[m_top], map);
		m_top++;
		Copy_Position(Ps[p_top].tmp_pos, position, Ps[p_top].tmp_pend, p_end);
		p_top++;
		//����
		Play(position[p_end - 1][0], position[p_end - 1][1], turn);
		//�ݹ� 
		value = -Alpha_Beta(nn,map, -1 * beta, -1 * alpha, -1 * turn, depth - 1);
		//���� 
		m_top -= 1;
		Copy_Board(map, tmp_map[m_top]);
		p_top -= 1;
		Copy_Position(position, Ps[p_top].tmp_pos, p_end, Ps[p_top].tmp_pend);
		//��� 
		if (value > alpha)
		{
			if (value >= beta)
				return value;
			alpha = value > alpha ? value : alpha;
		}
		if (value > maxx)
		{
			maxx = value;
			//��¼����ƶ� 
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
	Check_Neighbor(best_x, best_y, map, turn);//�ҵ���� 
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
		while (1)//ѵ��ģʽ��ʹ���������� 
		{

			Output();
			int flag;
			flag = CanMove(map, turn);//����Ƿ��п����岽 
			if (flag == 0)
			{
				no_more++;
				if (no_more == 2) break; //˫����������� 
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
	else//ѵ��ģʽ 
	{
		//int t;
		Start(1);
		printf("Input Training times\n");
		//scanf("%d", &t);//����ѵ����������
		ifstream fin("input.txt");
		ofstream fout("output.txt");
		int count = 0;
		while (!fin.eof())
		{
			count++;
			Start(0);
			if (fin)
			{//���뱾��ѵ����
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
				//i �� j���� �ڰ������
				for(int i = 0 ; i < CANDIDATE ;i ++ )
					for(int j = i+1 ; j < CANDIDATE ; j ++ )
					{
						int Coin = GetRand(100);
						//����ʱ i���ַ���j����
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
							flag = CanMove(map, turn);//����Ƿ��п����岽 
							if (flag == 0)
							{
								no_more++;
								if (no_more == 2) break; //˫����������� 
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
				//����ѡ�񽻻�
				Selection();
				for (int i = 0; i < OFFSPRING - 2; i += 2)
				{
					Combination(ANN[i], ANN[i + 1], ANN[CANDIDATE - i - 2], ANN[CANDIDATE - i - 1]);
					Mutation(ANN[CANDIDATE - i - 2]);
					Mutation(ANN[CANDIDATE - i - 1]);
				}
			}
		}
		//д�뽫finalд���ļ�
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