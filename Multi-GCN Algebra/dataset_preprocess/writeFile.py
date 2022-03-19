t0 = '{"id": "1",\n"context": ["0昨天联系过了 退换货", "1这边等到仓库收到商品，质检后，就会给您安排退款或者换货流程了，我们也是会催促仓库尽快处理的呢，请您耐心等待。", "0算了，我退了在买吧", "1您可以自己下单呢"],\n"behavior": ["下单"],\n"object": ["休闲鞋"],\n"state": [["none", "object"],["none","number"],["none","style"],["none","price"]],\n"Class": "order"},\n\n'
t1 = '{"id": "2",\n"context": ["0昨天联系过了 退换货", "1这边等到仓库收到商品，质检后，就会给您安排退款或者换货流程了，我们也是会催促仓库尽快处理的呢，请您耐心等待。", "0算了，我退了在买吧", "1您可以自己下单呢"],\n"behavior": ["退"],\n"object": ["休闲鞋"],\n"state": [["object", "none"],["number", "none"],["money", "none"],["none", "reason"]],\n"Class": "return"},\n\n'
t2 = ' {"id": "4",\n"context": ["0亲，我才刚刚收到打开就是穿过的", "1真的非常抱歉，可能仓库发货的时候没有注意呢，我们商品支持14天无理由退换货的哟"],\n"behavior": ["换"],\n"object": ["休闲鞋"],\n"state": [["none", "reason"],["origin","color"],["origin","style"],["origin","size"],["none","origin"]],\n"Class": "exchange"},\n\n'
t3 = '{"id": "5",\n"context": ["0那衣服我也申请退了 要确认收货才能退货是吧？ ", "1是的 这边等待仓库收到货才能退款哈"],\n"behavior": ["收"],\n"object": ["T恤"],\n"state": [["none","address"],["none","people"],["none","phonenum"],["none","object"],["object","confirm"],["confirm","comment"]],\n"Class": "receive"},\n\n'
t4 = '{"id": "7",\n"context":["1您好，只有已付款会为您保留库存的，建议您及时拍下订单后尽快付款，以免没有库存，成功付款后，仓库也会尽快给您安排寄出的","0好的"],\n"behavior": ["付款"],\n"object": ["板鞋"],\n"state": [["none", "object"],["object","price"],["price","account"],["account","bank"]],\n"Class": "payment"}, \n\n'
t5 = '{"id": "8",\n"context":["0啥时候发货呀","1耐心等待哦"],\n"behavior": ["发货"],\n"object": ["跑步鞋"],\n"state": [["none", "object"],["object","people"],["people","address"],["people","phonenum"]],\n"Class": "deliver"}, \n\n'
behavior_list = [t0,t1,t2,t3,t4,t5]

s = open('sample1200.txt', 'w', encoding='utf-8')
for i in range(200):
    for j in range(6):
        s.write(behavior_list[j])
s.close()

