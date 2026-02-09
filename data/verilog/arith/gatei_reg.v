`timescale 1ns/1ps
module gatei #(
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
  reg [DATA_TYPE - 1 : 0] reg_outs;

  always @(posedge clk) begin
    if (rst) begin
      reg_outs <= {DATA_TYPE{1'b0}};
    end else if (ins_valid) begin
      reg_outs <= ins;
    end
  end

  assign outs = ins_valid ? ins : reg_outs;
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule
