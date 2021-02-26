import heapq

li = [(1,3, [[1], [0]]), (1, 2, [[2], [0]]), (1,1, [[3], [0]])]

# using heapify to convert list into heap
heapq.heapify(li)

# printing created heap
print("The created heap is : ", end="")
print(list(li))

# using heappush() to push elements into heap
# pushes 4
heapq.heappush(li, (3, 1, [[2], [0]]))

# printing modified heap
print("The modified heap after push is : ", end="")
print(list(li))

# using heappop() to pop smallest element
print("The popped and smallest element is : ", end="")
for i in range(len(li)):
    print(heapq.heappop(li))

print(len(li), li)
