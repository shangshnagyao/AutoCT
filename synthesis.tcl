# synthesize.tcl - Design Compiler synthesis script

# 设置库和目标技术
set target_library "/home/ICer/PycharmProjects/PythonProject/gscl45nm.db "
set synthetic_library    "/home/ICer/PycharmProjects/PythonProject/gscl45nm.db "
set link_library "* $target_library"
# set symbol_library "/home/ic_libs/TSMC.90/aci/sc-x/symbols/synopsys/tsmc090.sdb"

# 读取设计文件
read_file -format verilog {temp_multiplier.v}

# 设置顶层模块
current_design multiplier

# 链接设计
link

set_max_area 0
# 设置时钟约束
set_max_delay -from [all_inputs] -to [all_outputs] 10

# 设置负载和驱动
# set_load 0.05 [all_outputs]
# set_driving_cell -lib_cell INVX1 [all_inputs]
# set_structure false
# set_flatten false
# set compile_enable_optimization false
# set compile_eliminate_redundant_logic false
# set_dont_touch [current_design]
# 综合设计
compile  -incremental

# 生成报告
report_area > dc_report_area.txt
report_timing > dc_report_timing.txt

# 退出
quit
