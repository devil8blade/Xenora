// SPDX-License-Identifier: GPL-3.0-or-later

use xenora_shader::{Architecture, CompileError, CompileOptions, compile, compile_with};

const FIXTURE: &[u8] = include_bytes!("../../../fixtures/rdna2/add.bin");

#[test]
fn compiles_original_fixture() {
    let result = compile(Architecture::Gfx1030, FIXTURE).unwrap();
    assert_eq!(result.ir, include_str!("../../../fixtures/golden/add.ir"));
    assert_eq!(
        result.msl,
        include_str!("../../../fixtures/golden/add.metal")
    );
}

#[test]
fn load_wait_is_required_before_using_loaded_data() {
    let mut code = FIXTURE.to_vec();
    code.drain(8..12);
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify { offset: 16, .. })
    ));
}

#[test]
fn store_wait_is_required_before_termination() {
    let mut code = FIXTURE.to_vec();
    code.drain(36..40);
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify { offset: 36, .. })
    ));
}

#[test]
fn rejects_unaligned_length_before_decoding() {
    assert!(matches!(
        compile(Architecture::Gfx1030, &FIXTURE[..FIXTURE.len() - 1]),
        Err(CompileError::InvalidLength { .. })
    ));
}

const BRANCH: &[u8] = include_bytes!("../../../fixtures/rdna2/bitwise-branch.bin");
const FLOAT: &[u8] = include_bytes!("../../../fixtures/rdna2/float-exact.bin");

#[test]
fn uniform_branch_selects_each_known_path() {
    let taken = compile(Architecture::Gfx1030, BRANCH).unwrap();
    assert_eq!(
        taken.ir,
        include_str!("../../../fixtures/golden/bitwise-branch.ir")
    );
    assert_eq!(
        taken.msl,
        include_str!("../../../fixtures/golden/bitwise-branch.metal")
    );

    let mut equal = BRANCH.to_vec();
    equal[17] = 0x87; // Compare s8 with 7 instead of 6.
    let fallthrough = compile(Architecture::Gfx1030, &equal).unwrap();
    assert!(fallthrough.ir.contains("branch.uniform 0x0020 taken=false"));
    assert!(!fallthrough.ir.contains("and.b32"));
    assert!(fallthrough.ir.contains("const.u32 0x00000000"));
}

#[test]
fn rejects_branch_into_a_literal_word() {
    let mut invalid = BRANCH.to_vec();
    invalid[20] = 3; // Branch to byte 36, inside the literal move at byte 32.
    assert!(matches!(
        compile(Architecture::Gfx1030, &invalid),
        Err(CompileError::Decode(error)) if error.offset == 20
    ));
}

#[test]
fn compiles_only_proven_exact_float_operations() {
    let result = compile(Architecture::Gfx1030, FLOAT).unwrap();
    assert_eq!(
        result.ir,
        include_str!("../../../fixtures/golden/float-exact.ir")
    );
    assert_eq!(
        result.msl,
        include_str!("../../../fixtures/golden/float-exact.metal")
    );

    let mut invalid = FLOAT.to_vec();
    invalid[16..20].copy_from_slice(&0x007f_ffffu32.to_le_bytes());
    assert!(matches!(
        compile(Architecture::Gfx1030, &invalid),
        Err(CompileError::Verify { offset: 40, .. })
    ));
}

#[test]
fn enforces_the_fixture_size_limit_before_decoding() {
    assert!(matches!(
        compile(Architecture::Gfx1030, &[]),
        Err(CompileError::InvalidLength { bytes: 0 })
    ));
    let mut code = FIXTURE.to_vec();
    // Prepend redundant s_waitcnt vmcnt(0) instructions to reach 1024 bytes.
    code.splice(
        ..0,
        [0x70, 0x3f, 0x8c, 0xbf].repeat((1024 - FIXTURE.len()) / 4),
    );
    assert!(compile(Architecture::Gfx1030, &code).is_ok());
    code.extend_from_slice(&[0; 4]);
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::InvalidLength { bytes: 1028 })
    ));
}

#[test]
fn preserves_the_decoder_diagnostic_and_error_source() {
    let error = compile(Architecture::Gfx1030, &[0xff; 4]).unwrap_err();
    let expected = "gfx1030 decode at byte 0x0: unsupported opcode in 0xffffffff";
    assert_eq!(error.to_string(), expected);
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        expected
    );
}

#[test]
fn rejects_register_moves_into_index_and_outstanding_load_destinations() {
    for (offset, instruction) in [
        (12, [0x01, 0x03, 0x00, 0x7e]), // v_mov_b32_e32 v0, v1
        (24, [0x02, 0x03, 0x02, 0x7e]), // v_mov_b32_e32 v1, v2
    ] {
        let mut code = FIXTURE.to_vec();
        if offset == 24 {
            // Move the load wait after the register move. v2 is initialized.
            code.drain(8..12);
            code.splice(16..16, instruction);
        } else {
            code.splice(offset..offset, instruction);
        }
        let expected_offset = if offset == 24 { 16 } else { offset };
        assert!(matches!(
            compile(Architecture::Gfx1030, &code),
            Err(CompileError::Verify {
                offset,
                reason: "move writes the index register or replaces a pending load",
            }) if offset == expected_offset
        ));
    }
}

#[test]
fn refuses_to_terminate_with_an_outstanding_load() {
    let mut code = FIXTURE.to_vec();
    // Add a second input load after the completed store. An output exists,
    // but the new load still needs its own wait.
    code.splice(40..40, FIXTURE[..8].iter().copied());
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify {
            offset: 48,
            reason: "termination requires one completed store and no pending loads",
        })
    ));
}

#[test]
fn protects_reserved_scalar_resource_registers() {
    let mut code = FIXTURE.to_vec();
    code.splice(..0, 0xbe87_0380u32.to_le_bytes()); // s_mov_b32 s7, 0
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify {
            offset: 0,
            reason: "scalar move would modify a resource descriptor",
        })
    ));
    code[2] = 0x88; // s8 is outside both resource descriptors.
    assert!(compile(Architecture::Gfx1030, &code).is_ok());
}

#[test]
fn proves_a_float_product_using_multiplication() {
    let mut code: Vec<_> = [
        0x7e02_02ffu32, // v_mov_b32_e32 v1, 0x4e800000 (2^30)
        0x4e80_0000,
        0x7e04_02ff, // v_mov_b32_e32 v2, 0x40000000 (2.0)
        0x4000_0000,
        0x1006_0501, // v_mul_f32_e32 v3, v1, v2
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .collect();
    code.extend_from_slice(&FIXTURE[28..]);
    code[25] = 3; // Store v3.
    // The product 2^31 is exact, while 2^30 + 2 is not representable.
    let shader = compile(Architecture::Gfx1030, &code).unwrap();
    assert!(shader.ir.contains("mul.f32 t0, t1"));
    assert!(
        shader
            .msl
            .contains("as_type<float>(t0) * as_type<float>(t1)")
    );
}

#[test]
fn rejects_load_into_index_or_before_waiting_on_previous_load() {
    let mut code = FIXTURE.to_vec();
    code[5] = 0; // Load into reserved v0.
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify { offset: 0, .. })
    ));
    let mut code = FIXTURE.to_vec();
    code.splice(8..8, FIXTURE[..8].iter().copied());
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify { offset: 8, .. })
    ));
}

#[test]
fn checks_fixture_layout_and_bindings_even_in_skipped_code() {
    for (prefix, expected_offset, expected_reason) in [
        (
            &[0xbf82_0001u32, 0xbf81_0000][..], // Skip an early s_endpgm.
            8,
            "instruction follows program termination",
        ),
        (
            &[0xbf82_0002, 0xe030_2000, 0x8001_0100], // Skip load from s[4:7].
            4,
            "unsupported resource or buffer index",
        ),
        (
            &[0xbf82_0002, 0xe070_2000, 0x8000_0100], // Skip store to s[0:3].
            4,
            "unsupported resource or buffer index",
        ),
    ] {
        let code: Vec<_> = prefix
            .iter()
            .copied()
            .flat_map(u32::to_le_bytes)
            .chain(FIXTURE.iter().copied())
            .collect();
        assert!(matches!(
            compile(Architecture::Gfx1030, &code),
            Err(CompileError::Verify { offset, reason })
                if offset == expected_offset && reason == expected_reason
        ));
    }

    // A structurally valid instruction on the skipped path need not read an
    // initialized register. Here v_mov_b32_e32 v1, v255 never runs.
    let code: Vec<_> = [0xbf82_0001u32, 0x7e02_03ff]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(FIXTURE.iter().copied())
        .collect();
    let compiled = compile(Architecture::Gfx1030, &code).unwrap();
    assert!(compiled.ir.contains("branch.uniform 0x0008 taken=true"));
    assert!(compiled.msl.contains("uint t1 = input[index];"));
    assert!(compiled.msl.contains("output[index] = t3;"));
}

#[test]
fn scalar_alu_results_reach_the_kernel_as_constants() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-alu.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-alu.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-alu.metal")
    );
}

#[test]
fn scalar_logic_results_reach_the_kernel_as_constants() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-logic.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-logic.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-logic.metal")
    );
}

#[test]
fn scalar_compare_results_resolve_every_branch() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-compare.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-compare.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-compare.metal")
    );
}

#[test]
fn scalar_immediate_results_resolve_every_branch() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-immediate.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-immediate.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-immediate.metal")
    );
}

#[test]
fn scalar_unary_results_reach_the_kernel_as_constants() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-unary.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-unary.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-unary.metal")
    );
}

#[test]
fn scalar_arith_results_reach_the_kernel_as_constants() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-arith.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-arith.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-arith.metal")
    );
}

#[test]
fn vector_integer_forms_lower_to_per_lane_templates() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-integer.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/vector-integer.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/vector-integer.metal")
    );
}

#[test]
fn vector_f_and_t_compares_lower_to_operand_free_wave_masks() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-compare-ft.bin");
    let result = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        result.ir,
        include_str!("../../../fixtures/golden/vector-compare-ft.ir")
    );
    assert_eq!(
        result.msl,
        include_str!("../../../fixtures/golden/vector-compare-ft.metal")
    );
}

#[test]
fn vector_compares_lower_to_per_thread_wave_masks_and_selects() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-compare.bin");
    let result = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        result.ir,
        include_str!("../../../fixtures/golden/vector-compare.ir")
    );
    assert_eq!(
        result.msl,
        include_str!("../../../fixtures/golden/vector-compare.metal")
    );
}

#[test]
fn vop1_integer_fixture_lowers_unary_and_masked_swap() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-unary.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/vector-unary.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/vector-unary.metal")
    );
}

/// load; mov v2,immA; mov v3,immB; and v2,v2,v3; xor v1,v1,v2; store.
/// Independently: A & B = 0x12005600.
fn constant_and_fixture() -> Vec<u8> {
    const A: u32 = 0x1234_5678;
    const B: u32 = 0xff00_ff00;
    [
        0xe030_2000,
        0x8000_0100, // buffer_load_dword v1
        0xbf8c_3f70, // s_waitcnt vmcnt(0)
        0x7e04_02ff,
        A, // v_mov_b32 v2, A
        0x7e06_02ff,
        B,           // v_mov_b32 v3, B
        0x3604_0702, // v_and_b32 v2, v2, v3
        0x3a02_0501, // v_xor_b32 v1, v1, v2
        0xe070_2000,
        0x8001_0100, // buffer_store_dword v1
        0xbbfd_0000, // s_waitcnt_vscnt
        0xbf81_0000, // s_endpgm
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .collect()
}

#[test]
fn default_compile_matches_fold_constants_off() {
    let code = constant_and_fixture();
    let baseline = compile(Architecture::Gfx1030, &code).unwrap();
    let explicit = compile_with(
        Architecture::Gfx1030,
        &code,
        CompileOptions {
            fold_constants: false,
        },
    )
    .unwrap();
    assert_eq!(baseline.ir, explicit.ir);
    assert_eq!(baseline.msl, explicit.msl);
    assert!(baseline.ir.contains("and.b32"));
}

#[test]
fn opt_in_constant_folding_rewrites_pure_vector_and() {
    let code = constant_and_fixture();
    let plain = compile_with(
        Architecture::Gfx1030,
        &code,
        CompileOptions {
            fold_constants: false,
        },
    )
    .unwrap();
    let folded = compile_with(
        Architecture::Gfx1030,
        &code,
        CompileOptions {
            fold_constants: true,
        },
    )
    .unwrap();

    // Independent expectation: 0x12345678 & 0xff00ff00 == 0x12005600.
    assert!(
        plain.ir.contains("and.b32"),
        "unoptimized IR should keep the vector and: {}",
        plain.ir
    );
    assert!(
        !folded.ir.contains("and.b32"),
        "folded IR should drop the pure and: {}",
        folded.ir
    );
    assert!(
        folded.ir.contains("const.u32 0x12005600"),
        "folded IR should materialize the independent constant: {}",
        folded.ir
    );
    assert!(
        folded.msl.contains("0x12005600u"),
        "folded MSL should use the independent constant: {}",
        folded.msl
    );
}

#[test]
fn folding_is_noop_on_add_fixture() {
    let folded = compile_with(
        Architecture::Gfx1030,
        FIXTURE,
        CompileOptions {
            fold_constants: true,
        },
    )
    .unwrap();
    let plain = compile(Architecture::Gfx1030, FIXTURE).unwrap();
    // add.bin has no pure constant-constant vector op, so folding is a no-op.
    assert_eq!(folded.ir, plain.ir);
    assert_eq!(folded.msl, plain.msl);
    assert_eq!(plain.ir, include_str!("../../../fixtures/golden/add.ir"));
}
