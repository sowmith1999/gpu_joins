## Introduction:

Joins on the GPU, The goal of the project was to support Relational Algebra operation Join(**⋈**) on the GPU and use that operation to do Transitive Closure and Intersection of graphs on the GPU. To efficiently support these operations, we use Hash Indexed Sorted Array[1] (HISA) structure. Supporting Relation Algebra on the GPU, makes way to accelerating Program Analysis, Graph Mining and several other industrial workloads.

The rest of the document explains about 
- Joins
	- What is a Join, an Example.
	- What are different types of Joins.
- Importance to the field
	- What does an efficient Join enable us to do.
- CPU design
	- How does Souffle Joins work?
- HISA(Hash Indexed Sorted Array)
	- How it works?
	- Advantages of using HISA
- Graph Intersection on the GPU
	- What is Graph Intersection?
	- Kernel design
	- Benchmarks
- TC
	- What is TC
	- Kernel Design
	- Benchmarks
- Conclusion

## Joins
Here in the context of this project, Join directly correlates to a SQL Inner Join, joining two tables on one or more columns. **⋈** is the Relational Algebra Operator for Join. Join is described in the literature as "Two relations can also be joined into one on a subset of columns they have in common. Join combines two relations into one, where a subset of columns are required to have matching values, and generalizes both intersection and Cartesian product operations."[2]

  ![[Join.svg]]

Two Relations Path and Edge, Joined the column z, result in a new relation path_new.

#### Homogeneous Joins
When two or more tables are being Joined on the same Column.
TODO: Add line or two to explain, what is happening here, may be put the equation of Join. Mention Graph Intersection as an example.

![[HomogenousJoin.svg]]

### Heterogeneous Join
When two or more relations are being joined on different Columns.
TODO: Add line or two to explain, what is happening here, may be put the equation of Join. Mention SG as an example.


![[HeterogeneousJoin.svg]]

TODO: Talk about why Hetero is hard to parallelize and how souffle does it, Loop Joins.



## Importance to the field
Supporting Relational Algebra primitives like Join on the GPU lets us support wide range of analytics work loads in graphs, machine learning, program analysis and formal verification.[2]
Especially in Datalog, Join is a ubiquitous operation in Datalog. 

## CPU Design
Souffle is a CPU based Parallel Datalog Compiler and Engine, It takes datalog rules and convert them into performant parallel C++ code. Their backend supports very efficient joins, They use B-Tree like structure  

TODO: Add more about how they do load balancing between threads and why it doesn't scale. and one code snippet of loop join.

## Hash Indexed Sorted Array
This data structure is proposed in GDlog[1]. It is at the center of supporting efficient operations on GPU. From the data structure we are expecting, 
- **Efficient range Querying**: This enables us each thread assigned to outer relation to make range queries for matching column values in the inner relation and is important to avoid linear scan of the inner relation.
- **Parallel Insertion**: A Relation should be able to support insertions from different threads in parallel.
- **Multiple Join Columns**: Support indexing over multiple columns, to support multi column joins.
- **Efficient Deduplication**: To be used in datalog fixed point loop, we need to be able to check and figure if a iteration has any tuples that are not duplicates of the full relation accumulated over previous iterations.
In our Implementation of HISA, we only support single join column and only uint32 type for keys and values. 
HISA has three components:
- **Map** - An Open Addressing hash table with linear probing: Is the part that supports efficient indexing into the relation.
- **Sorted Array** - A flat array storing the offsets of the rows, and is sorted across all columns, sorted array helps support range queries.
- **Data Array** - A flat array that store the tuples in the relation in a row major fashion.

![[HISA.svg]]

## Graph Intersection on GPU

TODO: Graphs intersection on GPU

## Transitive Closure on GPU


$$f(R) = R \cup \{(x,z) \mid (x,y) \in R \wedge (y,z) \in R\}
$$

## References
1. Sun, Y., Shovon, A. R., Gilray, T., Micinski, K., & Kumar, S. (2023). GDlog: A GPU-Accelerated Deductive Engine. _arXiv preprint arXiv:2311.02206_.
2. Gilray, T., & Kumar, S. (2019, December). Distributed relational algebra at scale. In _2019 IEEE 26th International Conference on High Performance Computing, Data, and Analytics (HiPC)_ (pp. 12-22). IEEE.
3. 

