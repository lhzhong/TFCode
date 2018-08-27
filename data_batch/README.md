## Tensorflow数据队列输入

1. 调用 tf.train.slice_input_producer，从 本地文件里抽取tensor，准备放入Filename Queue（文件名队列）中;  
2. 调用 tf.train.batch，从文件名队列中提取tensor，使用单个或多个线程，准备放入文件队列;  
3. 调用 tf.train.Coordinator() 来创建一个线程协调器，用来管理之后在Session中启动的所有线程;  
4. 调用tf.train.start_queue_runners, 启动入队线程，由多个或单个线程，按照设定规则，把文件读入Filename Queue中。函数返回线程ID的列表，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）;  
5. 文件从 Filename Queue中读入内存队列的操作不用手动执行，由tf自动完成;  
6. 调用sess.run 来启动数据出列和执行计算;  
7. 使用 coord.should_stop()来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，会抛出一个 OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程了;  
8. 使用coord.request_stop()来发出终止所有线程的命令，使用coord.join(threads)把线程加入主线程，等待threads结束。
> 注：如果再tf.train.slice_input_producer 函数中设置了 num_epochs 的数量， 文件队列末尾有结束标志，读到这个结束标志的时候抛出 OutofRangeError 异常，就可以结束各个线程了。 但是如果不设置 num_epochs 的数量，则文件队列是无限循环的，没有结束标志，程序会一直执行下去。
