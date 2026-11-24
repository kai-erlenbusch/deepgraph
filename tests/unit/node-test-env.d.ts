declare module 'node:test' {
    export function describe(name: string, fn: () => void | Promise<void>): void;
    export function it(name: string, fn: () => void | Promise<void>): void;
}

declare module 'node:assert/strict' {
    function assert(value: unknown, message?: string | Error): asserts value;
    namespace assert {
        function equal<T>(actual: unknown, expected: T, message?: string | Error): asserts actual is T;
        function strictEqual<T>(actual: unknown, expected: T, message?: string | Error): asserts actual is T;
        function deepEqual<T>(actual: unknown, expected: T, message?: string | Error): asserts actual is T;
        function ok(value: unknown, message?: string | Error): asserts value;
        function throws(block: () => unknown, error?: RegExp | ((err: unknown) => boolean), message?: string): void;
        function rejects(
            block: (() => Promise<unknown>) | Promise<unknown>,
            error?: RegExp | ((err: unknown) => boolean),
            message?: string
        ): Promise<void>;
    }
    export default assert;
}
