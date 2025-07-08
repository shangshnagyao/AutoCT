# synthesize.tcl - Design Compiler synthesis script

set target_library "/home/ICer/PycharmProjects/PythonProject/gscl45nm.db "
set synthetic_library    "/home/ICer/PycharmProjects/PythonProject/gscl45nm.db "
set link_library "* $target_library"

read_file -format verilog {temp_multiplier.v}

current_design multiplier

link

set_max_area 0
set_max_delay -from [all_inputs] -to [all_outputs] 10

# set_load 0.05 [all_outputs]
# set_driving_cell -lib_cell INVX1 [all_inputs]
# set_structure false
# set_flatten false
# set compile_enable_optimization false
# set compile_eliminate_redundant_logic false
# set_dont_touch [current_design]
compile 

report_area > dc_report_area.txt
report_timing > dc_report_timing.txt

quit
