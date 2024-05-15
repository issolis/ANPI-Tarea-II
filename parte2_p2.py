import numpy as np
import sympy as sp
from sympy import exp, symbols, sin, cos, log, E, lambdify, SympifyError

def getMulVec(matrix,  i, j): 
    vec = []

    for k in range (i, len(matrix)): 
        vec = vec + [matrix[k][j]]
    return vec

def putOneInRow(vec, i): 
    return [p/vec[i] for p in vec]
def putZerosInColumn(vec_1, vec_2, i):
    return [-1 * vec_1[k]* vec_2[i] + vec_2[k] for k in range(len(vec_1))] 

def makeUpperTriangular(matrix, vec): 
    if np.linalg.det(np.array(matrix)) == 0: 
        return 0
    tam = len(matrix[0])
    for k in range(tam): 
        matrix[k] = matrix [k] + [vec[k]]
    for i in range (tam - 1): 

        matrix[i] = putOneInRow(matrix[i], i)
        for k in range(i + 1, tam):
            matrix[k] = putZerosInColumn(matrix[i], matrix[k],  i)
        
    for k in range (tam): 
        vec [k] = matrix[k][tam]
        matrix[k] = matrix [k][0:tam]
    return [(matrix), vec]

def sustAtras (mat:np.array, indVec): 
    lenRow= mat.shape[0]
    sol =([0]*(lenRow))
    sol[lenRow - 1] = indVec[lenRow-1]/mat [lenRow-1,lenRow-1]
    counter = lenRow - 2
    while counter >= 0:
        sum = 0
        for j in range (counter+1, lenRow): 
            sum += mat[counter,j]*sol[j]
        sol [counter] = (indVec[counter] - sum)/mat[counter,counter]
        counter-=1
    return (sol)

def convertToFunction(expression, syms):
    syms = symbols(syms)
    return lambdify(syms, expression)

def jacobiana(f, vec): 
    tam = len(f)
    tam1 = len(vec)
    mat = [[0 for _ in range(tam)] for _ in range(tam)]
    
    for i in range(tam): 
        for j in range(tam1):
            mat[i][j] = sp.diff(sp.sympify(f[i]), vec[j])
    
    return mat

f_f = ['x**2 + y**2 +z**2 -1', '2*x**2 + y**2 - 4*z', '3*x**2 - 4*y + z**2']

def evaluateF(f, vec, vecVar):
    return convertToFunction(f,vecVar)(*vec)

def evaluateJacobiana(jacobiana, vec, variables): 
    k = 0
    for i in jacobiana: 
        k+=1
    return [ [evaluateF(jacobiana[i][j], vec, variables) for j in range(len(vec))] for i in range (k)]

def newton(x, f, variables, tol, iterMax): 
    j_f = jacobiana(f, variables)
    for i in range (iterMax): 
        f_k = evaluateF(f, x, variables)
        error = np.linalg.norm(np.array(f_k))
        if (error) < tol:
            return { "Solution ": x, "Iterations " : i , "Error ": error }
        matrixUP = makeUpperTriangular(evaluateJacobiana(j_f, x, variables), f_k)
        j_k =np.array( matrixUP[0] )
        y = sustAtras(j_k, matrixUP[1])
        x = (np.array(x) - np.array(y)).tolist() 
    return {"Solution ": x, "Iterations " : i + 1, "error ": np.linalg.norm(np.array(evaluateF(f, x, variables)))}





