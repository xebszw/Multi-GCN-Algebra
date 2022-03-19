node_id = 0
graph_id = 0
node_num = 0
graph_list = [0,0,0,0,0,0]
comment = '-4.691781997680664, 0.3991755545139313'
size =  '1.6928369998931885, 3.7349300384521484'
phonenum = '-1.6516047716140747, 0.9707381725311279'
style = '0.2511957287788391, 0.6446484327316284'
bank  = '-5.330697059631348, 0.17957967519760132'
people = '-0.5757513642311096, 0.6193016171455383'
price = '-1.6635115146636963, 0.20378421247005463'
none = '-0.42770498991012573, 0.6686766147613525'
money = '-0.4862912595272064, 1.6471513509750366'
reason = '0.9712159037590027, 2.697937488555908'
account = '-1.7088524103164673, -0.2498539835214615'
color = '2.743896961212158, 4.829883575439453'
number = '-0.5619820356369019, 1.1011254787445068'
confirm = '-2.8724658489227295, 0.26746562123298645'
address = '-0.47131869196891785, 0.709621787071228'
origin = '1.072642207145691, 2.1139986515045166'
object = '-1.6540405750274658, 0.40867191553115845'


s = open('sample1200.txt', 'r', encoding='utf-8')
f0 = open('BEHAVIOR/BEHAVIOR_A.txt', 'w')
f1 = open('BEHAVIOR/BEHAVIOR_graph_indicator.txt', 'w')
f2 = open('BEHAVIOR/BEHAVIOR_graph_labels.txt', 'w')
f3 = open('BEHAVIOR/BEHAVIOR_node_attributes.txt', 'w')
f4 = open('BEHAVIOR/BEHAVIOR_node_labels.txt', 'w')


for line in s.readlines():
    #if 'id' in line:
     #   print(line)

    if 'behavior' in line and '单' in line:
        graph_list[0]+=1
        graph_id += 1
        node_num = 5
        f0.write(str(node_id + 1) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 1) + ', ' + str(node_id + 3) + '\n' +
                 str(node_id + 1) + ', ' + str(node_id + 4) + '\n' +
                 str(node_id + 1) + ', ' + str(node_id + 5) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 3) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 1)+ '\n' +
                 str(node_id + 5) + ', ' + str(node_id + 1) + '\n')
        node_id += node_num
        for i in range(node_num):
            f1.write(str(graph_id) + '\n')
        f2.write('1\n')
        f3.write( none + '\n' +
                  object + '\n' +
                  number + '\n' +
                  style + '\n' +
                  price + '\n')
        #f4.write('0\n1\n1\n1\n1\n')
        f4.write('1\n2\n3\n4\n5\n')

    elif 'behavior' in line and '退' in line:
        graph_list[1] += 1
        graph_id += 1
        node_num = 5
        f0.write(str(node_id + 1) + ', ' + str(node_id + 4) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 4) + '\n' +
                 str(node_id + 3) + ', ' + str(node_id + 4) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 5) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 3) + '\n' +
                 str(node_id + 5) + ', ' + str(node_id + 4) + '\n')
        node_id += node_num
        for i in range(node_num):
            f1.write(str(graph_id) + '\n')
        f2.write('2\n')
        f3.write( object + '\n' +
                  number + '\n' +
                  money + '\n' +
                  none + '\n' +
                  reason + '\n')
        #f4.write('1\n1\n1\n0\n1\n')
        f4.write('2\n3\n6\n1\n7\n')

    elif 'behavior' in line and '换' in line:
        graph_list[2] += 1
        graph_id += 1
        node_num = 6
        f0.write(str(node_id + 1) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 1) + ', ' + str(node_id + 3) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 4) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 5) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 6) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 3) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 5) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 6) + ', ' + str(node_id + 2) + '\n')
        node_id += node_num
        for i in range(node_num):
            f1.write(str(graph_id) + '\n')
        f2.write('3\n')
        f3.write(none + '\n' +
                 origin + '\n' +
                 reason + '\n' +
                 color + '\n' +
                 style + '\n' +
                 size + '\n')
        #f4.write('0\n1\n1\n1\n1\n1\n')
        f4.write('1\n8\n7\n9\n4\n10\n')

    elif 'behavior' in line and '收' in line:
        graph_list[3] += 1
        graph_id += 1
        node_num = 7
        f0.write(str(node_id + 1) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 1) + ', ' + str(node_id + 3) + '\n' +
                 str(node_id + 1) + ', ' + str(node_id + 4) + '\n' +
                 str(node_id + 1) + ', ' + str(node_id + 5) + '\n' +
                 str(node_id + 5) + ', ' + str(node_id + 6) + '\n' +
                 str(node_id + 6) + ', ' + str(node_id + 7) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 3) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 5) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 6) + ', ' + str(node_id + 5) + '\n' +
                 str(node_id + 7) + ', ' + str(node_id + 6) + '\n')
        node_id += node_num
        for i in range(node_num):
            f1.write(str(graph_id) + '\n')
        f2.write('4\n')
        f3.write(none + '\n' +
                 address + '\n' +
                 people + '\n' +
                 phonenum + '\n' +
                 object + '\n' +
                 confirm + '\n' +
                 comment + '\n')
        #f4.write('0\n1\n1\n1\n1\n1\n1\n')
        f4.write('1\n11\n12\n13\n2\n14\n15\n')

    elif 'behavior' in line and ('买' in line or '购' in line or '拍' in line or '付' in line):
        graph_list[4] += 1
        graph_id += 1
        node_num = 5
        f0.write(str(node_id + 1) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 3) + '\n' +
                 str(node_id + 3) + ', ' + str(node_id + 4) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 5) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 3) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 3) + '\n' +
                 str(node_id + 5) + ', ' + str(node_id + 4) + '\n')
        node_id += node_num
        for i in range(node_num):
            f1.write(str(graph_id) + '\n')
        f2.write('5\n')
        f3.write( none + '\n' +
                  object + '\n' +
                  price + '\n' +
                  account + '\n' +
                  bank + '\n')
        #f4.write('0\n1\n1\n1\n1\n')
        f4.write('1\n2\n5\n16\n17\n')

    elif 'behavior' in line and ('寄' in line or '发' in line):
        graph_list[5] += 1
        graph_id += 1
        node_num = 5
        f0.write(str(node_id + 1) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 3) + '\n' +
                 str(node_id + 3) + ', ' + str(node_id + 4) + '\n' +
                 str(node_id + 3) + ', ' + str(node_id + 5) + '\n' +
                 str(node_id + 2) + ', ' + str(node_id + 1) + '\n' +
                 str(node_id + 3) + ', ' + str(node_id + 2) + '\n' +
                 str(node_id + 4) + ', ' + str(node_id + 3) + '\n' +
                 str(node_id + 5) + ', ' + str(node_id + 3) + '\n')
        node_id += node_num
        for i in range(node_num):
            f1.write(str(graph_id) + '\n')
        f2.write('6\n')
        f3.write( none + '\n' +
                  object + '\n' +
                  people + '\n' +
                  address + '\n' +
                  phonenum + '\n')
        #f4.write('0\n1\n1\n1\n1\n')
        f4.write('1\n2\n12\n11\n13\n')

print(graph_id)
print(graph_list)
f0.close()
f1.close()
f2.close()
f3.close()
f4.close()
s.close()