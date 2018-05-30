#!/usr/bin/env python

# https://soooprmx.com/archives/5845

import re
import sys


'''
import re
pat = r'(?:'          # 캡쳐하지 않는 그룹, 여기서는 조건문처럼 쓰인다.
    r'(?<=[^\d\.])' # look backward로 숫자나 점이 아닌 문자가 왼쪽에 있고
    r'(?=\d)'       # 오른쪽은 숫자가 있는 지점
    r'|(?=[^\d\.])'  # 만약, 숫자나 점 다음이라면, 그 다음은 숫자나 소수점이 아니어야 한다.
    r')'
regexp = re.compile(pat)
'''

def tokenize(expStr):
    pat = re.compile(r'(?:(?<=[^\d\.])(?=\d)|(?=[^\d\.]))', re.MULTILINE)
    return [x for x in re.sub(pat, ' ', expStr).split(' ') if x]

def parse_expr(expStr):
    tokens = tokenize(expStr)
    op = dict(zip('*/+-()', (50, 50, 40, 40, 0, 0)))
    output = []
    stack = []

    for item in tokens:
        print (item)
        if item not in op:
            output.append(item)
        elif item == '(':
            stack.append(item)
        elif item == ')':
            while stack != [] and \
                      stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack != [] and \
                    op[stack[-1]] > op[item]:
                output.append(stack.pop())
            stack.append(item)

    while stack:
        output.append(stack.pop())

    return output

def calc_expr(tokens):
    operations = {
        '*': lambda x, y: y * x,
        '/': lambda x, y: y / x,
        '+': lambda x, y: y + x,
        '-': lambda x, y: y - x
    }

    stack = []

    for item in tokens:
        if item not in operations:
            if '.' in item:
                stack.append(float(item))
            else:
                stack.append(int(item))
        else:
            x = stack.pop()
            y = stack.pop()
            stack.append(operations[item](x, y))
    return stack[-1]


def process(expStr):
    parse_expr(expStr)
    #print(calc_expr(parse_expr(expStr)))


def main():
    if len(sys.argv) > 1:
        x = ' '.join(sys.argv[1:])
    else:
        #x = input()
        #x = (((sasdfg OR "asdf's" OR asdf OR asdf OR asdf OR asdf ) AND (asdf OR "asdf's" OR "asdf 9" OR ))~10 OR asdf OR asdf9 )'
        x = "(((sasd1 OR asdf2 OR asd3f OR as4df OR a5sdf ) AND (a6sdf OR a7sdf))~10 OR a8sdf OR as9df9 )"
    process(x)


if __name__ == '__main__':
    main()


