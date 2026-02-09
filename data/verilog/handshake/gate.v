`timescale 1ns/1ps
module gate #(
  parameter DATA_TYPE = 32
) (
  // inputs
  input clk,
  input rst,
  input [DATA_TYPE - 1 : 0] ins,
  input ins_valid,
  input outs_ready,

  // outputs
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);

  assign outs = ins_valid ? ins : {DATA_TYPE{1'b0}};
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule
