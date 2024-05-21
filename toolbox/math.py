# %% from sympy import symbols (import)
from sympy import symbols

# %% The function searches for positive integer arithmetic sequences that meet the given conditions and prints the first n terms.
def find_arithmetic_term(term_index_1, term_index_2, product, print_term):
    """
    该函数寻找满足给定条件的正整数等差数列并打印前n项。

    Args:
        term_index_1 (int): 第一个用于计算乘积的项的索引位置。
        term_index_2 (int): 第二个用于计算乘积的项的索引位置。
        product (int): 期望得到的目标乘积。
        print_term (int): 指定打印等差数列到哪一项。

    Example:
        find_arithmetic_term(1, 3, 5, 5)  这代表寻找一个等差数列，使得第一项和第三项的乘积为5，并打印到第五项。
    Returns:
        int: 返回要打印的项的值.
    """
    for a_1 in range(1, 6):
        for d in range(1, 6):
            t1 = a_1 + (term_index_1 - 1) * d
            t2 = a_1 + (term_index_2 - 1) * d
            if t1 * t2 == product:
                print_term_value = a_1 + (print_term - 1) * d
                print("公差为", d, "的等差数列的前", print_term, "项为：", end=" ")
                for i in range(1, print_term + 1):
                    print(a_1 + (i - 1) * d, end=" ")
                print()
                return print_term_value

# %% from sympy import solve (import)
from sympy import solve

# %% import math (import)
import math