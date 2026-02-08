import json


def main():
    """
    Python中使用JSON主要完成：
    1. 将Python字典、列表转换为JSON字符串
    2. 读取JSON字符串，转换为Python字典或列表
    
    主要使用Python内置的json库：
    - json.dumps(): 将字典或列表转换为JSON字符串
    - json.loads(): 将JSON字符串转换为Python字典或列表
    """
    
    print("=" * 60)
    print("Python JSON 使用示例")
    print("=" * 60)
    
    # ========== 示例1: 使用 json.dumps() 将Python对象转换为JSON字符串 ==========
    print("\n【示例1】使用 json.dumps() 将Python字典转换为JSON字符串")
    print("-" * 60)
    
    # 创建一个包含中文的Python字典
    python_dict = {
        "姓名": "张三",
        "年龄": 25,
        "城市": "北京",
        "职业": "软件工程师",
        "技能": ["Python", "JavaScript", "Java"],
        "联系方式": {
            "电话": "13800138000",
            "邮箱": "zhangsan@example.com"
        }
    }
    
    print("原始Python字典:")
    print(python_dict)
    print(f"类型: {type(python_dict)}")
    
    # 使用 json.dumps() 转换为JSON字符串
    # ensure_ascii=False 确保中文能正常显示
    json_string = json.dumps(python_dict, ensure_ascii=False, indent=2)
    
    print("\n转换后的JSON字符串:")
    print(json_string)
    print(f"类型: {type(json_string)}")
    
    # ========== 示例2: 使用 json.dumps() 将Python列表转换为JSON字符串 ==========
    print("\n\n【示例2】使用 json.dumps() 将Python列表转换为JSON字符串")
    print("-" * 60)
    
    python_list = [
        {"产品": "笔记本电脑", "价格": 5999, "库存": 50},
        {"产品": "智能手机", "价格": 3999, "库存": 100},
        {"产品": "平板电脑", "价格": 2999, "库存": 30}
    ]
    
    print("原始Python列表:")
    print(python_list)
    print(f"类型: {type(python_list)}")
    
    json_string_list = json.dumps(python_list, ensure_ascii=False, indent=2)
    
    print("\n转换后的JSON字符串:")
    print(json_string_list)
    print(f"类型: {type(json_string_list)}")
    
    # ========== 示例3: 使用 json.loads() 将JSON字符串转换为Python对象 ==========
    print("\n\n【示例3】使用 json.loads() 将JSON字符串转换为Python字典")
    print("-" * 60)
    
    # 一个JSON格式的字符串
    json_str = '{"学校": "清华大学", "专业": "计算机科学", "年级": 3, "课程": ["数据结构", "算法设计", "操作系统"]}'
    
    print("原始JSON字符串:")
    print(json_str)
    print(f"类型: {type(json_str)}")
    
    # 使用 json.loads() 转换为Python字典
    python_dict_from_json = json.loads(json_str)
    
    print("\n转换后的Python字典:")
    print(python_dict_from_json)
    print(f"类型: {type(python_dict_from_json)}")
    
    # 访问字典中的值
    print(f"\n访问字典值:")
    print(f"学校: {python_dict_from_json['学校']}")
    print(f"专业: {python_dict_from_json['专业']}")
    print(f"课程: {python_dict_from_json['课程']}")
    
    # ========== 示例4: 使用 json.loads() 将JSON字符串转换为Python列表 ==========
    print("\n\n【示例4】使用 json.loads() 将JSON字符串转换为Python列表")
    print("-" * 60)
    
    json_list_str = '[{"名称": "苹果", "价格": 8.5}, {"名称": "香蕉", "价格": 6.0}, {"名称": "橙子", "价格": 7.2}]'
    
    print("原始JSON字符串:")
    print(json_list_str)
    print(f"类型: {type(json_list_str)}")
    
    python_list_from_json = json.loads(json_list_str)
    
    print("\n转换后的Python列表:")
    print(python_list_from_json)
    print(f"类型: {type(python_list_from_json)}")
    
    # 遍历列表
    print("\n遍历列表内容:")
    for item in python_list_from_json:
        print(f"  {item['名称']}: ¥{item['价格']}/斤")
    
    # ========== 示例5: 完整的数据转换流程 ==========
    print("\n\n【示例5】完整的数据转换流程：Python对象 -> JSON字符串 -> Python对象")
    print("-" * 60)
    
    # 原始Python数据
    original_data = {
        "订单号": "ORD20240101001",
        "客户信息": {
            "姓名": "李四",
            "地址": "上海市浦东新区"
        },
        "商品列表": [
            {"商品名": "商品A", "数量": 2, "单价": 100},
            {"商品名": "商品B", "数量": 1, "单价": 200}
        ],
        "总金额": 400
    }
    
    print("1. 原始Python字典:")
    print(json.dumps(original_data, ensure_ascii=False, indent=2))
    
    # 转换为JSON字符串（用于传输或存储）
    json_data = json.dumps(original_data, ensure_ascii=False)
    print(f"\n2. 转换为JSON字符串（用于传输或存储）:")
    print(json_data)
    
    # 从JSON字符串转换回Python对象（用于处理）
    restored_data = json.loads(json_data)
    print(f"\n3. 从JSON字符串转换回Python字典:")
    print(json.dumps(restored_data, ensure_ascii=False, indent=2))
    
    # 验证数据是否一致
    print(f"\n4. 验证数据一致性:")
    print(f"   原始数据 == 恢复数据: {original_data == restored_data}")
    
    # ========== 示例6: ensure_ascii 参数的作用对比 ==========
    print("\n\n【示例6】ensure_ascii 参数的作用对比")
    print("-" * 60)
    
    chinese_data = {"消息": "你好，世界！", "状态": "成功"}
    
    print("原始数据:", chinese_data)
    
    # ensure_ascii=True (默认值) - 中文会被转义为Unicode编码
    json_with_ascii = json.dumps(chinese_data, ensure_ascii=True)
    print(f"\nensure_ascii=True (默认):")
    print(json_with_ascii)
    
    # ensure_ascii=False - 中文正常显示
    json_without_ascii = json.dumps(chinese_data, ensure_ascii=False)
    print(f"\nensure_ascii=False (推荐用于中文):")
    print(json_without_ascii)
    
    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    print("1. json.dumps(字典或列表, ensure_ascii=False):")
    print("   - 将Python字典或列表转换为JSON字符串")
    print("   - ensure_ascii=False 确保中文能正常显示")
    print("   - 返回值: JSON字符串")
    print("\n2. json.loads(json字符串):")
    print("   - 将JSON字符串转换为Python字典或列表")
    print("   - 返回值: Python字典或Python列表")
    print("=" * 60)


if __name__ == "__main__":
    main()
