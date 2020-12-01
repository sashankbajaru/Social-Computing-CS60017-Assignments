import csv
import snap

amazon_main = open('com-amazon.ungraph.txt','r')
csv_reader = csv.reader(amazon_main,delimiter='\t')
amazon_sgraph = open('subgraphs/amazon.elist','w+')

node_cnt = 0
for node in csv_reader:
  if((int(node[0]))%4==0 and (int(node[1]))%4==0):
    node_cnt += 1
    amazon_sgraph.write(node[0]+'\t'+node[1]+'\n')

print(node_cnt)

fb_main = open('facebook_combined.txt','r')
csv_reader = csv.reader(fb_main,delimiter=' ')
fb_sgraph = open('subgraphs/facebook.elist','w+')

node_cnt = 0
for node in csv_reader:
  if((int(node[0]))%5==0 or (int(node[1]))%5==0):
    continue
  else:
    node_cnt += 1
    fb_sgraph.write(node[0]+'\t'+node[1]+'\n')

print(node_cnt)