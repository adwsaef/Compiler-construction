# Based on utils.py by Jeremy Siek
# Adapted by permission from https://github.com/IUCompilerCourse/python-student-support-code
# See EOC-LICENSE for details
import os
import sys
import ast
from ast import ( 
    Module, Expr, Pass, Assign, AnnAssign, Return, Name, Constant, Add, Sub, Mult, And, Or,
    USub, Not, UnaryOp, Call, If, IfExp, While, Compare, Eq, NotEq, Lt, LtE, Gt, GtE, Is, Tuple, Subscript, List, BinOp, BoolOp
)

from dataclasses import dataclass

indent_amount = 2

def indent_stmt():
    return " " * indent_amount


def indent():
    global indent_amount
    indent_amount += 2


def dedent():
    global indent_amount
    indent_amount -= 2

def label_name(n: str) -> str:
    if sys.platform == "darwin":
        return '_' + n
    else:
        return n

def strip_module(node:ast.AST) -> list[ast.stmt]:
   match node:
       case Module(body=body):
           return body
       case ast.stmt():
           return [node]
       case _:
           raise ValueError(f"Expected Module or stmt, got {type(node)}")

def get_first_expr(node: ast.AST) -> ast.expr:
    match node:
        case Module(body=[Expr(value=e),*rest]):
            return e
        case Expr(value=e):
            return e
        case _:
            raise ValueError(f"Expected Module or Expr, got {type(node)}")

def parse_stmts(source: str) -> list[ast.stmt]:
    """parse a string into a list of statements"""
    return strip_module(ast.parse(source))

def parse_expr(source: str) -> ast.expr:
    """parse a string into an expression"""
    return get_first_expr(ast.parse(source))

################################################################################
# Generating unique names
################################################################################

name_id = 0


def generate_name(name):
    global name_id
    ls = name.split('.')
    new_id = name_id
    name_id += 1
    return ls[0] + '.' + str(new_id)


def str_Module(self):
    indent()
    body = ''.join([str(s) for s in self.body])
    dedent()
    return body


Module.__str__ = str_Module


def repr_Module(self):
    return 'Module(' + repr(self.body) + ')'


Module.__repr__ = repr_Module


def str_Expr(self):
    return indent_stmt() + str(self.value) + '\n'


Expr.__str__ = str_Expr


def repr_Expr(self):
    return indent_stmt() + 'Expr(' + repr(self.value) + ')'


Expr.__repr__ = repr_Expr


def str_Pass(self):
    return indent_stmt() + 'pass' + '\n'


Pass.__str__ = str_Pass


def repr_Pass(self):
    return indent_stmt() + 'Pass()'


Pass.__repr__ = repr_Pass


def str_Assign(self):
    return indent_stmt() + str(self.targets[0]) + ' = ' + str(self.value) + '\n'


Assign.__str__ = str_Assign


def repr_Assign(self):
    return indent_stmt() + 'Assign(' + repr(self.targets) + ', ' + repr(self.value) + ')'


Assign.__repr__ = repr_Assign


def str_AnnAssign(self):
    return indent_stmt() + str(self.target) + ' : ' + str(self.annotation) + ' = ' + str(self.value) + '\n'


AnnAssign.__str__ = str_AnnAssign


def repr_AnnAssign(self):
    return indent_stmt() + 'AnnAssign(' + repr(self.target) + ', ' \
           + repr(self.annotation) + ', ' + repr(self.value) + repr(self.simple) + ')'


AnnAssign.__repr__ = repr_AnnAssign


def str_Return(self):
    return indent_stmt() + 'return ' + str(self.value) + '\n'


Return.__str__ = str_Return


def repr_Return(self):
    return indent_stmt() + 'Return(' + repr(self.value) + ')'


Return.__repr__ = repr_Return


def str_Name(self):
    if hasattr(self, 'has_type'):
        return self.id + ':' + str(self.has_type)
    else:
        return self.id


Name.__str__ = str_Name


def repr_Name(self):
    return 'Name(' + repr(self.id) + ')'


Name.__repr__ = repr_Name


def str_Constant(self):
    return str(self.value)


Constant.__str__ = str_Constant


def repr_Constant(self):
    return 'Constant(' + repr(self.value) + ')'


Constant.__repr__ = repr_Constant


def str_Add(self):
    return '+'


Add.__str__ = str_Add


def repr_Add(self):
    return 'Add()'


Add.__repr__ = repr_Add


def str_Sub(self):
    return '-'


Sub.__str__ = str_Sub


def repr_Sub(self):
    return 'Sub()'


Sub.__repr__ = repr_Sub


def str_Mult(self):
    return '*'


Mult.__str__ = str_Mult


def repr_Mult(self):
    return 'Mult()'


Mult.__repr__ = repr_Mult


def str_And(self):
    return 'and'


And.__str__ = str_And


def repr_And(self):
    return 'And()'


And.__repr__ = repr_And


def str_Or(self):
    return 'or'


Or.__str__ = str_Or


def repr_Or(self):
    return 'Or()'


Or.__repr__ = repr_Or


def str_BinOp(self):
    return '(' + str(self.left) + ' ' + str(self.op) + ' ' + str(self.right) + ')'


BinOp.__str__ = str_BinOp


def repr_BinOp(self):
    return 'BinOp(' + repr(self.left) + ', ' + repr(self.op) + ', ' + repr(self.right) + ')'


BinOp.__repr__ = repr_BinOp


def str_BoolOp(self):
    return '(' + str(self.values[0]) + ' ' + str(self.op) + ' ' + str(self.values[1]) + ')'


BoolOp.__str__ = str_BoolOp


def repr_BoolOp(self):
    return repr(self.values[0]) + ' ' + repr(self.op) + ' ' + repr(self.values[1])


BoolOp.__repr__ = repr_BoolOp


def str_USub(self):
    return '-'


USub.__str__ = str_USub


def repr_USub(self):
    return 'USub()'


USub.__repr__ = repr_USub


def str_Not(self):
    return 'not'


Not.__str__ = str_Not


def repr_Not(self):
    return 'Not()'


Not.__repr__ = repr_Not


def str_UnaryOp(self):
    return str(self.op) + '(' + str(self.operand) + ')'


UnaryOp.__str__ = str_UnaryOp


def repr_UnaryOp(self):
    return 'UnaryOp(' + repr(self.op) + ', ' + repr(self.operand) + ')'


UnaryOp.__repr__ = repr_UnaryOp


def str_Call(self):
    return str(self.func) \
           + '(' + ', '.join([str(arg) for arg in self.args]) + ')'


Call.__str__ = str_Call


def repr_Call(self):
    return 'Call(' + repr(self.func) + ', ' + repr(self.args) + ')'


Call.__repr__ = repr_Call


def str_If(self):
    header = indent_stmt() + 'if ' + str(self.test) + ':\n'
    indent()
    thn = ''.join([str(s) for s in self.body])
    els = ''.join([str(s) for s in self.orelse])
    dedent()
    return header + thn + indent_stmt() + 'else:\n' + els


If.__str__ = str_If


def repr_If(self):
    return 'If(' + repr(self.test) + ', ' + repr(self.body) + ', ' + repr(self.orelse) + ')'


If.__repr__ = repr_If


def str_IfExp(self):
    return '(' + str(self.body) + ' if ' + str(self.test) + \
           ' else ' + str(self.orelse) + ')'


IfExp.__str__ = str_IfExp


def repr_IfExp(self):
    return 'IfExp(' + repr(self.test) + ', ' + repr(self.body) + \
        ', ' + repr(self.orelse) + ')'


IfExp.__repr__ = repr_IfExp


def str_While(self):
    header = indent_stmt() + 'while ' + str(self.test) + ':\n'
    indent()
    body = ''.join([str(s) for s in self.body])
    dedent()
    return header + body


While.__str__ = str_While


def repr_While(self):
    return 'While(' + repr(self.test) + ', ' + repr(self.body) + ', ' + repr(self.orelse) + ')'


While.__repr__ = repr_While


def str_Compare(self):
    return str(self.left) + ' ' + str(self.ops[0]) + ' ' + str(self.comparators[0])


Compare.__str__ = str_Compare


def repr_Compare(self):
    return 'Compare(' + repr(self.left) + ', ' + repr(self.ops) + ', ' \
           + repr(self.comparators) + ')'


Compare.__repr__ = repr_Compare


def str_Eq(self):
    return '=='


Eq.__str__ = str_Eq


def repr_Eq(self):
    return 'Eq()'


Eq.__repr__ = repr_Eq


def str_NotEq(self):
    return '!='


NotEq.__str__ = str_NotEq


def repr_NotEq(self):
    return 'NotEq()'


NotEq.__repr__ = repr_NotEq


def str_Lt(self):
    return '<'


Lt.__str__ = str_Lt


def repr_Lt(self):
    return 'Lt()'


Lt.__repr__ = repr_Lt


def str_LtE(self):
    return '<='


LtE.__str__ = str_LtE


def repr_LtE(self):
    return 'LtE()'


LtE.__repr__ = repr_LtE


def str_Gt(self):
    return '>'


Gt.__str__ = str_Gt


def repr_Gt(self):
    return 'Gt()'


Gt.__repr__ = repr_Gt


def str_GtE(self):
    return '>='


GtE.__str__ = str_GtE


def repr_GtE(self):
    return 'GtE()'


GtE.__repr__ = repr_GtE


def str_Is(self):
    return 'is'


Is.__str__ = str_Is


def repr_Is(self):
    return 'Is()'


Is.__repr__ = repr_Is


def str_Tuple(self):
    return '(' + ', '.join([str(e) for e in self.elts]) + ',)'


Tuple.__str__ = str_Tuple


def repr_Tuple(self):
    return 'Tuple(' + repr(self.elts) + ')'


Tuple.__repr__ = repr_Tuple


def str_List(self):
    return '[' + ', '.join([str(e) for e in self.elts]) + ']'


ast.List.__str__ = str_List


def repr_List(self):
    return 'List(' + repr(self.elts) + ')'


ast.List.__repr__ = repr_List


def str_Subscript(self):
    return str(self.value) + '[' + str(self.slice) + ']'


Subscript.__str__ = str_Subscript


def repr_Subscript(self):
    return 'Subscript(' + repr(self.value) + ', ' + repr(self.slice) \
           + ', ' + repr(self.ctx) + ')'


Subscript.__repr__ = repr_Subscript